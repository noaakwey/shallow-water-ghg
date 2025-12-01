/*******************************************************
 * GPP, RECO AND NEE FLUX MAPPING
 * Shallow Waters of Kuibyshev Reservoir
 *
 * This script applies calibrated regression models to map
 * carbon fluxes across the entire shallow water zone:
 *   - GPP: Gross Primary Production
 *   - Reco: Ecosystem Respiration
 *   - NEE: Net Ecosystem Exchange (Reco - GPP)
 *
 * Model specifications:
 *   GPP_11am = α₀ + α₁×LST_mean + α₂×MTCI + α₃×AWEInsh + α₄×LST_max² + α₅×(AWEInsh×LST_mean)
 *   Reco_11am = β₀ + β₁×LST_max + β₂×AWEInsh² + β₃×(NDWI×LST_mean) + β₄×(AWEInsh×LST_mean)
 *
 * Outputs:
 *   - Instantaneous_Fluxes_11am_umol_s.tif
 *   - Annual_Fluxes_Rate_umol_s.tif
 *   - Annual_Fluxes_Sum_umol_year.tif
 *   - Shallow_Water_Mask_2023.tif
 *
 * GEE Link: https://code.earthengine.google.com/9c94d46927cd229008d0d210a8c7eaa9
 *******************************************************/

//------------------------------------------------------
// 0. LIBRARIES
//------------------------------------------------------

var satLib = require('users/ortho/satellite-processing:satellite-processing');
var SatelliteProcessor = satLib.SatelliteProcessor;
var s2CloudProb = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

// Region of interest (import as asset or define inline)
var roi = reservoirPolygon;

//------------------------------------------------------
// 1. MODEL COEFFICIENTS (from Python calibration)
//------------------------------------------------------

// Reco coefficients
var RECO_INTERCEPT        = 2.4613052754;
var RECO_LST_MAX          = -0.0622376677;
var RECO_AWEINSH2         = -2.578172689920999e-07;
var RECO_NDWI_LST_MEAN    = 0.3947233035;
var RECO_AWEINSH_LST_MEAN = 4.450890880868273e-05;

// GPP coefficients
var GPP_INTERCEPT        = 22.5300861527;
var GPP_LST_MEAN         = -0.7930209460;
var GPP_MTCI             = -0.1112698148;
var GPP_AWEINSH          = -0.0070017170;
var GPP_LST_MAX2         = -0.0025788303;
var GPP_AWEINSH_LST_MEAN = 2.933430053528863e-04;

// Predictor clamp ranges (from training data)
var LST_MEAN_MIN = 5.109855; var LST_MEAN_MAX = 25.345112;
var LST_MAX_MIN = 5.474990;  var LST_MAX_MAX = 27.377663;
var MTCI_MIN = -4.151326;    var MTCI_MAX = 4.831204;
var NDWI_MIN = -0.066937;    var NDWI_MAX = 0.096783;
var AWEINSH_MIN = 2470.0516; var AWEINSH_MAX = 4748.9063;

// Clamp function to constrain predictors to training domain
function clampPredictors(img) {
  return img
    .addBands(img.select('MTCI').clamp(MTCI_MIN, MTCI_MAX), null, true)
    .addBands(img.select('AWEInsh').clamp(AWEINSH_MIN, AWEINSH_MAX), null, true)
    .addBands(img.select('NDWI').clamp(NDWI_MIN, NDWI_MAX), null, true);
}

//------------------------------------------------------
// 2. SCALING PARAMETERS
//------------------------------------------------------

var F_GPP  = 0.904;           // Diurnal correction: 11am → daily
var F_RECO = 0.978;           // Diurnal correction: 11am → daily
var DAYLIGHT_HOURS = 12;      // Average daylight hours
var FULL_DAY_HOURS = 24;      // Full day
var ICE_FREE_DAYS  = 200;     // Ice-free season (April 20 - November 5)
var SECONDS_DAYLIGHT = DAYLIGHT_HOURS * 3600;
var SECONDS_FULL_DAY = FULL_DAY_HOURS * 3600;

//------------------------------------------------------
// 3. TIME PERIOD
//------------------------------------------------------

var START_DATE_2025 = '2025-04-15';
var END_DATE_2025   = '2025-11-01';

// Reference period for shallow water mask
var SUMMER_START = '2023-06-01'; var SUMMER_END = '2023-08-31';
var SPRING_START = '2023-04-15'; var SPRING_END = '2023-05-15';
var AUTUMN_START = '2023-10-01'; var AUTUMN_END = '2023-10-31';

// Water detection thresholds
var MNDWI_PERMANENT_WATER = 0.0;
var MNDWI_SEASONAL_WATER  = -0.1;

// Cloud filtering
var LOCAL_CLOUD_PROB_THRESHOLD = 40;
var MAX_LOCAL_CLOUD_FRACTION   = 0.20;
var WINDOW_DAYS = 5;

// Export settings
var EXPORT_FOLDER = 'Carbon_NEE_Final';
var EXPORT_SCALE  = 10;
var EXPORT_CRS    = 'EPSG:32639';

//------------------------------------------------------
// 4. SENTINEL-2 PREPARATION
//------------------------------------------------------

var indicesWater = ['CIG', 'CVI', 'MTCI', 'NDWI', 'MNDWI', 'AWEInsh', 
                    'AWEIsh', 'NDVI', 'EVI2', 'WDRVI', 'SAVI', 'NMDI', 
                    'NDMI', 'GNDVI'];

function maskEdgesS2(img) {
  var b8aMask = img.select('B8A').mask();
  var b9Mask  = img.select('B9').mask();
  return img.updateMask(b8aMask).updateMask(b9Mask);
}

function addIndicesS2(img) {
  var withIdx = SatelliteProcessor.addIndices.universal(
    img, SatelliteProcessor.SENTINEL2, indicesWater, true);
  return maskEdgesS2(withIdx).resample('bilinear');
}

function addCloudMask(img) {
  var date = ee.Date(img.get('system:time_start'));
  var cloudProb = img.select('cloud_prob');
  var isCloud   = cloudProb.gt(LOCAL_CLOUD_PROB_THRESHOLD);
  var masked = img.updateMask(isCloud.not());
  var cloudFrac = isCloud.reduceRegion({
    reducer: ee.Reducer.mean(), 
    geometry: roi, 
    scale: 100, 
    maxPixels: 1e9, 
    bestEffort: true
  }).get('cloud_prob');
  return masked.set({
    'local_cloud_fraction': ee.Number(cloudFrac), 
    'date': date.format('YYYY-MM-dd')
  });
}

//------------------------------------------------------
// 5. SHALLOW WATER MASK (PHENOLOGICAL)
//------------------------------------------------------

// Summer permanent water reference
var s2Summer = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(SUMMER_START, SUMMER_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(addIndicesS2);
var permanentWater = s2Summer.median().select('MNDWI').gt(MNDWI_PERMANENT_WATER);

// Spring flooding extent
var s2Spring = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(SPRING_START, SPRING_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(addIndicesS2);
var springWater = s2Spring.median().select('MNDWI')
  .gt(MNDWI_SEASONAL_WATER)
  .updateMask(permanentWater);

// Autumn water extent
var s2Autumn = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(AUTUMN_START, AUTUMN_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(addIndicesS2);
var autumnWater = s2Autumn.median().select('MNDWI')
  .gt(MNDWI_SEASONAL_WATER)
  .updateMask(permanentWater);

// Shallow water: flooded in spring but exposed in autumn
var shallowWaterMask = springWater.and(autumnWater.not())
  .clip(roi)
  .rename('shallow_mask');

//------------------------------------------------------
// 6. PREPARE 2025 DATA
//------------------------------------------------------

var s2Raw2025 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(START_DATE_2025, END_DATE_2025);

var s2WithIdx2025 = s2Raw2025.map(addIndicesS2);

var innerJoin = ee.Join.inner();
var cpFilter = ee.Filter.equals({
  leftField: 'system:index', 
  rightField: 'system:index'
});

var s2Joined2025 = ee.ImageCollection(
  innerJoin.apply(s2WithIdx2025, s2CloudProb, cpFilter)
    .map(function(feat) {
      return ee.Image(feat.get('primary'))
        .addBands(ee.Image(feat.get('secondary'))
          .select('probability')
          .rename('cloud_prob'));
    })
);

var s2Clean2025 = s2Joined2025
  .map(addCloudMask)
  .filter(ee.Filter.lte('local_cloud_fraction', MAX_LOCAL_CLOUD_FRACTION));

//------------------------------------------------------
// 7. LANDSAT LST WITH SINE FALLBACK
//------------------------------------------------------

var landsatAll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(roi);

function preprocessLandsat(img) {
  var lstC = img.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST');
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1<<3).eq(0)
    .and(qa.bitwiseAnd(1<<4).eq(0));
  return img.addBands(lstC).updateMask(mask);
}

var landsatPre = landsatAll.map(preprocessLandsat);

// Sine model for LST when Landsat unavailable
var TWO_PI = 2 * Math.PI;

function lstSineMean(doy) { 
  var angle = doy.multiply(TWO_PI).divide(365); 
  return ee.Number(7.3225)
    .subtract(ee.Number(2.6157).multiply(angle.sin()))
    .subtract(ee.Number(17.4224).multiply(angle.cos())); 
}

function lstSineMax(doy) { 
  var angle = doy.multiply(TWO_PI).divide(365); 
  return ee.Number(10.9092)
    .subtract(ee.Number(0.0278).multiply(angle.sin()))
    .subtract(ee.Number(16.1691).multiply(angle.cos())); 
}

function getLSTfromLandsatOrSine(date) {
  var doy = date.getRelative('day', 'year').add(1);
  var subset = landsatPre.filterDate(
    date.advance(-WINDOW_DAYS, 'day'), 
    date.advance(WINDOW_DAYS, 'day')
  );
  var hasLandsat = subset.size().gt(0);
  
  var closest = ee.Image(ee.Algorithms.If(
    hasLandsat, 
    subset.map(function(ls) {
      return ls.set('dt', ee.Number(ls.get('system:time_start'))
        .subtract(date.millis()).abs());
    }).sort('dt').first(), 
    null
  ));
  
  var stats = ee.Algorithms.If(
    hasLandsat, 
    closest.select('LST').reduceRegion({
      reducer: ee.Reducer.mean().combine(ee.Reducer.max(), '', true), 
      geometry: roi, 
      scale: 30, 
      maxPixels: 1e9, 
      bestEffort: true
    }), 
    ee.Dictionary({})
  );
  
  return {
    LST_mean: ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(ee.Dictionary(stats).get('LST_mean'), null), 
      lstSineMean(doy), 
      ee.Dictionary(stats).get('LST_mean')
    )),
    LST_max: ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(ee.Dictionary(stats).get('LST_max'), null), 
      lstSineMax(doy), 
      ee.Dictionary(stats).get('LST_max')
    ))
  };
}

//------------------------------------------------------
// 8. FLUX CALCULATION
//------------------------------------------------------

var fluxCollection = s2Clean2025.map(function (img) {
  var date = img.date();
  
  // Apply predictor clamping
  img = clampPredictors(img);

  var MTCI = img.select('MTCI');
  var NDWI = img.select('NDWI');
  var AWEInsh = img.select('AWEInsh');
  
  var lstInfo  = getLSTfromLandsatOrSine(date);
  var LST_mean = ee.Image.constant(lstInfo.LST_mean)
    .clamp(LST_MEAN_MIN, LST_MEAN_MAX);
  var LST_max  = ee.Image.constant(lstInfo.LST_max)
    .clamp(LST_MAX_MIN, LST_MAX_MAX);

  // Derived terms
  var AWEInsh2 = AWEInsh.multiply(AWEInsh);
  var NDWI_LST_mean = NDWI.multiply(LST_mean);
  var AWEInsh_LST_mean = AWEInsh.multiply(LST_mean);
  var lstMax2 = LST_max.multiply(LST_max);

  // Reco at 11:00 AM
  var reco_11am = ee.Image.constant(RECO_INTERCEPT)
    .add(ee.Image.constant(RECO_LST_MAX).multiply(LST_max))
    .add(AWEInsh2.multiply(RECO_AWEINSH2))
    .add(NDWI_LST_mean.multiply(RECO_NDWI_LST_MEAN))
    .add(AWEInsh_LST_mean.multiply(RECO_AWEINSH_LST_MEAN))
    .max(0)  // Physical constraint: non-negative
    .rename('Reco_11am');

  // GPP at 11:00 AM
  var gpp_11am = ee.Image.constant(GPP_INTERCEPT)
    .add(ee.Image.constant(GPP_LST_MEAN).multiply(LST_mean))
    .add(MTCI.multiply(GPP_MTCI))
    .add(AWEInsh.multiply(GPP_AWEINSH))
    .add(ee.Image.constant(GPP_LST_MAX2).multiply(lstMax2))
    .add(AWEInsh_LST_mean.multiply(GPP_AWEINSH_LST_MEAN))
    .max(0)  // Physical constraint: non-negative
    .rename('GPP_11am');

  // Apply diurnal correction
  var reco_corrected = reco_11am.multiply(F_RECO).rename('Reco_corrected');
  var gpp_corrected = gpp_11am.multiply(F_GPP).rename('GPP_corrected');
  var nee_instant = reco_corrected.subtract(gpp_corrected).rename('NEE_instant');

  return img.addBands([reco_11am, gpp_11am, reco_corrected, gpp_corrected, nee_instant])
    .set({'date': date})
    .clip(roi);
});

//------------------------------------------------------
// 9. AGGREGATION (WITH SHALLOW WATER MASK)
//------------------------------------------------------

// For Monte Carlo input: mean instantaneous fluxes, masked
var reco_11am_mean = fluxCollection.select('Reco_11am').mean()
  .updateMask(shallowWaterMask)
  .rename('Reco_11am');

var gpp_11am_mean = fluxCollection.select('GPP_11am').mean()
  .updateMask(shallowWaterMask)
  .rename('GPP_11am');

// Seasonal means (masked)
var recoMean = fluxCollection.select('Reco_corrected').mean()
  .updateMask(shallowWaterMask).rename('Reco_mean');
var gppMean = fluxCollection.select('GPP_corrected').mean()
  .updateMask(shallowWaterMask).rename('GPP_mean');
var neeMean = fluxCollection.select('NEE_instant').mean()
  .updateMask(shallowWaterMask).rename('NEE_mean');

// Annual sums
var recoAnnualSum = recoMean.multiply(SECONDS_FULL_DAY)
  .multiply(ICE_FREE_DAYS).rename('Reco_annual_sum');
var gppAnnualSum  = gppMean.multiply(SECONDS_DAYLIGHT)
  .multiply(ICE_FREE_DAYS).rename('GPP_annual_sum');
var neeAnnualSum  = recoAnnualSum.subtract(gppAnnualSum)
  .rename('NEE_annual_sum');

var obsCount = fluxCollection.select('Reco_corrected').count()
  .updateMask(shallowWaterMask).rename('observation_count');

//------------------------------------------------------
// 10. VISUALIZATION
//------------------------------------------------------

Map.centerObject(roi, 10);
Map.addLayer(shallowWaterMask.selfMask(), {palette: ['cyan']}, 'Shallow Water Mask');
Map.addLayer(gpp_11am_mean, 
  {min: 0, max: 2.5, palette: ['white', 'green', 'red']}, 
  'GPP 11am (Shallow only)');
Map.addLayer(neeMean, 
  {min: -1, max: 2, palette: ['blue', 'white', 'red']}, 
  'NEE Mean');

//------------------------------------------------------
// 11. EXPORT
//------------------------------------------------------

function exportImage(img, desc, bandNames) {
  Export.image.toDrive({
    image: img.rename(bandNames),
    description: desc,
    folder: EXPORT_FOLDER,
    region: roi,
    scale: EXPORT_SCALE,
    crs: EXPORT_CRS,
    maxPixels: 1e12,
    shardSize: 256,
    formatOptions: {cloudOptimized: true}
  });
}

// Shallow water mask
exportImage(shallowWaterMask.unmask(0).toByte(), 
  'Shallow_Water_Mask_2023', ['shallow_mask']);

// Instantaneous fluxes (for MC analysis)
exportImage(ee.Image.cat([reco_11am_mean, gpp_11am_mean]), 
  'Instantaneous_Fluxes_11am_umol_s', ['Reco_11am', 'GPP_11am']);

// Seasonal mean flux rates
exportImage(ee.Image.cat([recoMean, gppMean, neeMean]), 
  'Annual_Fluxes_Rate_umol_s', ['Reco_mean', 'GPP_mean', 'NEE_mean']);

// Annual flux sums
exportImage(ee.Image.cat([recoAnnualSum, gppAnnualSum, neeAnnualSum]), 
  'Annual_Fluxes_Sum_umol_year', ['Reco_annual_sum', 'GPP_annual_sum', 'NEE_annual_sum']);

print('✅ Ready for Export');
