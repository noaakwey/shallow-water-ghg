/*******************************************************
 * SENTINEL-2 + LANDSAT LST PREDICTORS FOR EDDY COVARIANCE TOWER
 * Shallow Waters of Kuibyshev Reservoir
 *
 * This script extracts satellite predictors for flux model calibration:
 *   - Sentinel-2 SR optical bands and spectral indices
 *   - Landsat 8/9 Land Surface Temperature (LST)
 *   - Local cloud masking within tower footprint
 *
 * Outputs:
 *   1. S2_LandsatLST_Predictors_EddyCov_2025.csv - Main predictor table
 *   2. S2_Meta_EddyCov_LocalCloud_2025.csv - Scene metadata
 *
 * GEE Link: https://code.earthengine.google.com/e14ca393a297b6d6a2887907f2ede160
 *******************************************************/

//------------------------------------------------------
// 0. IMPORT LIBRARIES
//------------------------------------------------------

var satLib = require('users/ortho/satellite-processing:satellite-processing');
var SatelliteProcessor = satLib.SatelliteProcessor;

// Cloud probability collection for Sentinel-2
var s2CloudProb = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

//------------------------------------------------------
// 1. GEOMETRY: TOWER LOCATION AND FOOTPRINT
//------------------------------------------------------

// Eddy covariance tower location
var towerPoint = ee.Geometry.Point([49.28018703202701, 55.26938846171377]);

// Tower footprint (300 m buffer - simplified circle)
var footprint = towerPoint.buffer(300);

// Shallow water polygon (for visualization only)
var shallowWater = ee.Geometry.Polygon(
  [[[49.269749810478416, 55.273195593663495],
    [49.269749810478416, 55.264393579558295],
    [49.28562848784658, 55.264393579558295],
    [49.28562848784658, 55.273195593663495]]],
  null, false
);

// Region of interest for filtering
var roi = footprint;

//------------------------------------------------------
// 2. CONFIGURATION
//------------------------------------------------------

// Time period of interest
var startDate = '2025-04-01';
var endDate   = '2025-11-30';

// Local cloud fraction thresholds
var LOCAL_CLOUD_PROB_THRESHOLD = 40;   // Pixel is cloudy if prob > 40%
var MAX_LOCAL_CLOUD_FRACTION   = 0.20; // Max cloud fraction in footprint

// Landsat search window around S2 date (days)
var WINDOW_DAYS = 5;

// Export settings
var EXPORT_FOLDER = 'Carbon';
var EXPORT_SCALE  = 10;
var EXPORT_CRS    = 'EPSG:32639';

// Visualization toggle
var SHOW_MAP = false;

//------------------------------------------------------
// 3. SPECTRAL INDICES FOR WATER ECOSYSTEMS
//------------------------------------------------------

var indicesWater = [
  // Chlorophyll / Phytoplankton
  'CIG',
  'CVI',
  'MTCI',
  'TCI',

  // Water / Turbidity
  'NDWI',
  'MNDWI',
  'WI1', 'WI2',
  'AWEInsh',
  'AWEIsh',
  'S2WI',
  'SWM',
  'LSWI',

  // Macrophytes / Vegetation
  'NDVI',
  'EVI2',
  'WDRVI',
  'SAVI',

  // Moisture / Organics
  'NMDI',
  'NDMI',
  'GNDVI',
  'ARVI'
];

//------------------------------------------------------
// 4. SENTINEL-2 SR: BASE COLLECTION + INDICES
//------------------------------------------------------

print('================ SENTINEL-2 SR: BASE COLLECTION ================');

var s2Raw = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.calendarRange(4, 11, 'month')); // April-November

print('Total Sentinel-2 SR scenes in period:', s2Raw.size());

// Edge masking function
function maskEdgesS2(img) {
  var b8aMask = img.select('B8A').mask();
  var b9Mask  = img.select('B9').mask();
  return img.updateMask(b8aMask).updateMask(b9Mask);
}

// Add spectral indices using satellite-processing library
function addIndicesS2(img) {
  var withIdx = SatelliteProcessor.addIndices.universal(
    img,
    SatelliteProcessor.SENTINEL2,
    indicesWater,
    true  // skipUnsupported
  );
  // Edge mask and resample to 10 m
  withIdx = maskEdgesS2(withIdx).resample('bilinear');
  return withIdx;
}

var s2WithIdx = s2Raw.map(addIndicesS2);

print('After index calculation:', s2WithIdx.size());

//------------------------------------------------------
// 5. JOIN SENTINEL-2 WITH S2_CLOUD_PROBABILITY
//------------------------------------------------------

print('================ JOIN WITH S2_CLOUD_PROBABILITY ================');

var innerJoin = ee.Join.inner();
var cpFilter = ee.Filter.equals({
  leftField:  'system:index',
  rightField: 'system:index'
});

var s2Joined = ee.ImageCollection(
  innerJoin.apply(s2WithIdx, s2CloudProb, cpFilter)
    .map(function (feat) {
      var img = ee.Image(feat.get('primary'));
      var cp  = ee.Image(feat.get('secondary'))
        .select('probability')
        .rename('cloud_prob'); // 0-100 (%)
      return img.addBands(cp);
    })
);

print('After join with S2_CLOUD_PROBABILITY:', s2Joined.size());

//------------------------------------------------------
// 6. LOCAL CLOUD MASKING IN FOOTPRINT
//------------------------------------------------------

print('================ LOCAL CLOUD FRACTION CALCULATION ================');

var s2LocalCloud = s2Joined.map(function (img) {
  var date = ee.Date(img.get('system:time_start'));

  // Cloud/non-cloud pixel classification by threshold
  var cloudProb = img.select('cloud_prob'); // 0-100
  var isCloud   = cloudProb.gt(LOCAL_CLOUD_PROB_THRESHOLD);

  // Cloud fraction within footprint (0-1)
  var cloudFrac = isCloud.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: footprint,
    scale: 20,
    maxPixels: 1e7,
    bestEffort: true
  }).get('cloud_prob');

  var localCloudFrac = ee.Number(cloudFrac);

  // Mask cloudy pixels
  var clearMask = isCloud.not();
  var masked = img.updateMask(clearMask);

  // Time/date metadata
  var datetimeUTC   = date.format('YYYY-MM-dd HH:mm:ss');
  var timeUTC       = date.format('HH:mm:ss');
  var localTime     = date.advance(3, 'hour'); // MSK = UTC+3
  var datetimeLocal = localTime.format('YYYY-MM-dd HH:mm:ss');
  var timeLocal     = localTime.format('HH:mm:ss');
  var unixTimestamp = date.millis().divide(1000);
  var doy           = date.getRelative('day', 'year').add(1);

  // Scene metadata
  var solarZenith      = ee.Number(img.get('MEAN_SOLAR_ZENITH_ANGLE'));
  var solarAzimuth     = ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE'));
  var orbitNumber      = img.get('SENSING_ORBIT_NUMBER');
  var relativeOrbit    = img.get('RELATIVE_ORBIT_NUMBER');
  var orbitDirection   = img.get('SENSING_ORBIT_DIRECTION');
  var cloudCoverScene  = img.get('CLOUDY_PIXEL_PERCENTAGE');
  var systemIndex      = img.get('system:index');

  masked = masked.set({
    'date': date.format('YYYY-MM-dd'),
    'time_utc': timeUTC,
    'datetime_utc': datetimeUTC,
    'time_local': timeLocal,
    'datetime_local': datetimeLocal,
    'unix_timestamp': unixTimestamp,
    'doy': doy,
    'local_cloud_fraction': localCloudFrac,
    'cloud_cover_pct_scene': cloudCoverScene,
    'solar_zenith_deg': solarZenith,
    'solar_azimuth_deg': solarAzimuth,
    'orbit_number': orbitNumber,
    'relative_orbit': relativeOrbit,
    'orbit_direction': orbitDirection,
    'system_index': systemIndex
  });

  return masked;
});

print('Scenes with local cloud fraction calculated:', s2LocalCloud.size());

// Filter to "clean" scenes by local threshold
var s2Good = s2LocalCloud.filter(
  ee.Filter.lte('local_cloud_fraction', MAX_LOCAL_CLOUD_FRACTION)
);

print('Scenes with local_cloud_fraction <= ' + MAX_LOCAL_CLOUD_FRACTION + ':', s2Good.size());

//------------------------------------------------------
// 7. LANDSAT 8/9 L2: LST PREPARATION
//------------------------------------------------------

print('================ LANDSAT 8/9 L2: LST PREPARATION ================');

var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');

var landsatAll = l8.merge(l9).filterBounds(roi);

// Preprocessing: scaling, cloud masking, LST calculation
function preprocessLandsat(img) {
  // L2 scale factors
  var optical = img.select('SR_B.*').multiply(0.0000275).add(-0.2);
  var thermalK = img.select('ST_B10').multiply(0.00341802).add(149.0); // Kelvin

  // QA_PIXEL: bit 3 = cloud, bit 4 = shadow
  var qaPixel = img.select('QA_PIXEL');
  var cloudBitMask  = 1 << 3;
  var shadowBitMask = 1 << 4;
  var clear = qaPixel.bitwiseAnd(cloudBitMask).eq(0)
                     .and(qaPixel.bitwiseAnd(shadowBitMask).eq(0));

  // LST in Celsius
  var lstC = thermalK.subtract(273.15).rename('LST');

  return img.addBands(optical, null, true)
            .addBands(thermalK.rename('ST_B10_K'), null, true)
            .addBands(lstC)
            .updateMask(clear);
}

var landsatPre = landsatAll.map(preprocessLandsat);

print('Total Landsat 8/9 scenes in region:', landsatPre.size());

//------------------------------------------------------
// 8. EXTRACT PREDICTORS (S2 + LST) FROM FOOTPRINT
//------------------------------------------------------

print('================ PREDICTOR EXTRACTION (S2 + LST) ====================');

var predictorsFC = ee.FeatureCollection(
  s2Good.map(function (img) {
    var date = img.date();

    // --- 8.1 S2: Mean and std statistics within footprint ---
    var statsMean = ee.Dictionary(
      img.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: footprint,
        scale: EXPORT_SCALE,
        maxPixels: 1e13,
        bestEffort: true
      })
    );

    var statsStd = ee.Dictionary(
      img.reduceRegion({
        reducer: ee.Reducer.stdDev(),
        geometry: footprint,
        scale: EXPORT_SCALE,
        maxPixels: 1e13,
        bestEffort: true
      })
    );

    var keys      = statsMean.keys();
    var meanVals  = keys.map(function (k) { return statsMean.get(k); });
    var stdVals   = keys.map(function (k) { return statsStd.get(k); });
    var stdKeys   = keys.map(function (k) { return ee.String(k).cat('_std'); });

    var meanDict  = ee.Dictionary.fromLists(keys,    meanVals);
    var stdDict   = ee.Dictionary.fromLists(stdKeys, stdVals);
    var combined  = meanDict.combine(stdDict);

    // --- 8.2 Landsat LST for this S2 date ---
    var start = date.advance(-WINDOW_DAYS, 'day');
    var end   = date.advance(WINDOW_DAYS, 'day');

    var subset = landsatPre.filterDate(start, end);

    // Add time difference, select closest scene
    var withDiff = subset.map(function (ls) {
      var diff = ee.Number(ls.get('system:time_start'))
                    .subtract(date.millis())
                    .abs();
      return ls.set('time_diff', diff);
    });

    var hasLandsat = subset.size().gt(0);

    var closest = ee.Image(
      ee.Algorithms.If(
        hasLandsat,
        withDiff.sort('time_diff').first(),
        null
      )
    );

    // LST statistics within footprint
    var lstStats = ee.Dictionary(
      ee.Algorithms.If(
        hasLandsat,
        closest.select('LST').reduceRegion({
          reducer: ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), '', true)
            .combine(ee.Reducer.min(), '', true)
            .combine(ee.Reducer.max(), '', true),
          geometry: footprint,
          scale: 30,
          maxPixels: 1e13,
          bestEffort: true
        }),
        ee.Dictionary({})
      )
    );

    var hasLstStats = hasLandsat.and(lstStats.size().gt(0));

    // Landsat metadata (date and day difference)
    var landsatMeta = ee.Dictionary(
      ee.Algorithms.If(
        hasLstStats,
        ee.Dictionary({
          'landsat_date': ee.Date(closest.get('system:time_start'))
                            .format('YYYY-MM-dd'),
          'landsat_days_diff': ee.Number(closest.get('time_diff'))
                                .divide(86400000)
        }),
        ee.Dictionary({
          'landsat_date': null,
          'landsat_days_diff': null
        })
      )
    );

    // Merge LST statistics and metadata
    combined = combined
      .combine(lstStats, true)
      .combine(landsatMeta, true)
      .set('has_landsat', hasLstStats);

    // --- 8.3 Create Feature with S2 metadata ---
    var feature = ee.Feature(null, combined)
      .set({
        'date': img.get('date'),
        'datetime_utc': img.get('datetime_utc'),
        'time_utc': img.get('time_utc'),
        'datetime_local': img.get('datetime_local'),
        'time_local': img.get('time_local'),
        'unix_timestamp': img.get('unix_timestamp'),
        'doy': img.get('doy'),
        'local_cloud_fraction': img.get('local_cloud_fraction'),
        'cloud_cover_pct_scene': img.get('cloud_cover_pct_scene'),
        'solar_zenith_deg': img.get('solar_zenith_deg'),
        'solar_azimuth_deg': img.get('solar_azimuth_deg'),
        'orbit_number': img.get('orbit_number'),
        'relative_orbit': img.get('relative_orbit'),
        'orbit_direction': img.get('orbit_direction'),
        'system_index': img.get('system_index')
      });

    return feature;
  })
);

print('Total rows in predictor table (S2 + LST):', predictorsFC.size());
print('Sample records:', predictorsFC.limit(5));

//------------------------------------------------------
// 9. METADATA TABLE FOR ALL S2 SCENES (including cloudy)
//------------------------------------------------------

var metaFC = ee.FeatureCollection(
  s2LocalCloud.map(function (img) {
    return ee.Feature(null, {
      'date': img.get('date'),
      'datetime_utc': img.get('datetime_utc'),
      'time_utc': img.get('time_utc'),
      'datetime_local': img.get('datetime_local'),
      'time_local': img.get('time_local'),
      'unix_timestamp': img.get('unix_timestamp'),
      'doy': img.get('doy'),
      'local_cloud_fraction': img.get('local_cloud_fraction'),
      'cloud_cover_pct_scene': img.get('cloud_cover_pct_scene'),
      'solar_zenith_deg': img.get('solar_zenith_deg'),
      'solar_azimuth_deg': img.get('solar_azimuth_deg'),
      'orbit_number': img.get('orbit_number'),
      'relative_orbit': img.get('relative_orbit'),
      'orbit_direction': img.get('orbit_direction'),
      'system_index': img.get('system_index')
    });
  })
);

print('Total rows in S2 metadata table:', metaFC.size());

//------------------------------------------------------
// 10. EXPORT TO GOOGLE DRIVE
//------------------------------------------------------

// Main predictor table for regression modeling
Export.table.toDrive({
  collection: predictorsFC,
  description: 'S2_LandsatLST_Predictors_EddyCov_2025',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV'
});

// Metadata table for all S2 scenes
Export.table.toDrive({
  collection: metaFC,
  description: 'S2_Meta_EddyCov_LocalCloud_2025',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV'
});

print('Exports added to Tasks:');
print('  1) S2_LandsatLST_Predictors_EddyCov_2025');
print('  2) S2_Meta_EddyCov_LocalCloud_2025');

//------------------------------------------------------
// 11. VISUALIZATION (OPTIONAL)
//------------------------------------------------------

if (SHOW_MAP) {
  Map.centerObject(footprint, 13);

  var exampleAll = ee.Image(s2LocalCloud.first());
  Map.addLayer(
    exampleAll,
    {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
    'S2 RGB (all scenes)'
  );
  Map.addLayer(
    exampleAll.select('cloud_prob'),
    {min: 0, max: 100},
    'Cloud probability (0-100)'
  );

  var exampleGood = ee.Image(s2Good.first());
  Map.addLayer(
    exampleGood,
    {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
    'S2 RGB (cloud_frac <= ' + MAX_LOCAL_CLOUD_FRACTION + ')'
  );

  Map.addLayer(footprint,    {color: 'red'},    'EC Footprint (300 m)');
  Map.addLayer(shallowWater, {color: 'yellow'}, 'Shallow water template');
}
