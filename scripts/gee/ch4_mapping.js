/*******************************************************

- CH4 FLUX MODEL - BASED ON LANDSAT LST
- ═══════════════════════════════════════════════════════
- 
- QUADRATIC MODEL (calibrated on moment values):
- CH4_moment = 0.0053 - 0.00218×LST + 0.000212×LST²
- 
- PERFORMANCE:
- R² (LOO) = 0.466
- RMSE = 0.0260 µmol m⁻² s⁻¹
- n = 27 scenes
- 
- WORKFLOW:
- 1. Calculate CH4_moment from Landsat LST
- 1. Apply F_CH4 to get CH4_daily
- 1. Sum over ice-free period for annual
   *******************************************************/

//——————————————————
// 0. БИБЛИОТЕКИ И ROI
//——————————————————

var satLib = require("users/ortho/satellite-processing:satellite-processing");
var SatelliteProcessor = satLib.SatelliteProcessor;
var roi = reservoirPolygon;

//——————————————————
// 1. CH4 MODEL COEFFICIENTS
//——————————————————

// Quadratic model: CH4_moment = a + b×LST + c×LST²
var CH4_A = 0.005337;
var CH4_B = -0.002178;
var CH4_C = 0.00021193;

// LST clamp range (from training data)
var LST_MIN = 5.11;
var LST_MAX = 25.35;

// Diurnal correction (moment → daily)
var F_CH4 = 0.8761;

// Conversion: µmol m⁻² s⁻¹ → g CH4 m⁻² day⁻¹
var UMOL_TO_G_DAY = 1.3858;

// Ice-free period (same as GPP/Reco)
var ICE_FREE_DAYS = 200;  // April 20 - November 5

// GWP for CO2-equivalent
var GWP_CH4 = 28;

//——————————————————
// 2. LANDSAT LST COLLECTION (2025 season)
//——————————————————

var landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
.filterBounds(roi)
.filterDate("2025-05-01", "2025-10-15");

var landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
.filterBounds(roi)
.filterDate("2025-05-01", "2025-10-15");

var landsatAll = landsat8.merge(landsat9);

print("Landsat scenes:", landsatAll.size());

function preprocessLandsat(img) {
// Convert ST_B10 to Celsius
var lstC = img.select("ST_B10")
.multiply(0.00341802)
.add(149.0)
.subtract(273.15)
.rename("LST");

// Cloud/shadow mask
var qa = img.select("QA_PIXEL");
var cloudMask = qa.bitwiseAnd(1<<3).eq(0)
.and(qa.bitwiseAnd(1<<4).eq(0))
.and(qa.bitwiseAnd(1<<5).eq(0));

return lstC.updateMask(cloudMask)
.set("system:time_start", img.get("system:time_start"));
}

var landsatLST = landsatAll.map(preprocessLandsat);

//——————————————————
// 3. SHALLOW WATER MASK
//——————————————————

function addIndicesS2(img) {
return SatelliteProcessor.addIndices.universal(img, SatelliteProcessor.SENTINEL2, ["MNDWI"], true);
}

var s2Summer = ee.ImageCollection("COPERNICUS/S2_SR")
.filterBounds(roi)
.filterDate("2023-06-01", "2023-08-31")
.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
.map(addIndicesS2);
var permanentWater = s2Summer.median().select("MNDWI").gt(0.0);

var s2Spring = ee.ImageCollection("COPERNICUS/S2_SR")
.filterBounds(roi)
.filterDate("2023-04-15", "2023-05-15")
.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
.map(addIndicesS2);
var springWater = s2Spring.median().select("MNDWI").gt(-0.1).updateMask(permanentWater);

var s2Autumn = ee.ImageCollection("COPERNICUS/S2_SR")
.filterBounds(roi)
.filterDate("2023-10-01", "2023-10-31")
.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
.map(addIndicesS2);
var autumnWater = s2Autumn.median().select("MNDWI").gt(-0.1).updateMask(permanentWater);

var shallowWaterMask = springWater.and(autumnWater.not()).clip(roi).rename("shallow_mask");

//——————————————————
// 4. CALCULATE CH4 FOR EACH LANDSAT SCENE
//——————————————————

function calculateCH4(lstImg) {
// Clamp LST
var lstClamped = lstImg.clamp(LST_MIN, LST_MAX);

// CH4_moment = a + b×LST + c×LST²
var ch4Moment = ee.Image.constant(CH4_A)
.add(lstClamped.multiply(CH4_B))
.add(lstClamped.pow(2).multiply(CH4_C))
.max(0)
.rename("CH4_moment");

// CH4_daily = CH4_moment × F_CH4
var ch4Daily = ch4Moment.multiply(F_CH4).rename("CH4_daily");

return ee.Image.cat([lstClamped.rename("LST"), ch4Moment, ch4Daily])
.updateMask(shallowWaterMask)
.set("system:time_start", lstImg.get("system:time_start"));
}

var ch4Collection = landsatLST.map(calculateCH4);

print("CH4 images:", ch4Collection.size());

//——————————————————
// 5. AGGREGATE TO SEASONAL MEAN
//——————————————————

// Mean LST over season
var lstMean = ch4Collection.select("LST").mean().rename("LST_mean");

// Mean CH4_moment (for MC input)
var ch4MomentMean = ch4Collection.select("CH4_moment").mean().rename("CH4_moment_mean");

// Mean CH4_daily
var ch4DailyMean = ch4Collection.select("CH4_daily").mean().rename("CH4_daily_mean");

// Observation count
var obsCount = ch4Collection.select("LST").count().rename("obs_count");

// Annual CH4 (g m⁻² yr⁻¹)
var ch4Annual = ch4DailyMean.multiply(UMOL_TO_G_DAY).multiply(ICE_FREE_DAYS).rename("CH4_g_m2_yr");

// CO2-equivalent
var ch4CO2eq = ch4Annual.multiply(GWP_CH4).rename("CH4_CO2eq_g_m2_yr");

//——————————————————
// 6. STATISTICS
//——————————————————

print("═══════════════════════════════════════════════════════");
print("CH4 MODEL (LST-BASED) RESULTS");
print("═══════════════════════════════════════════════════════");

// LST stats
var lstStats = lstMean.reduceRegion({
reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
geometry: shallowWaterMask.geometry(),
scale: 30,
maxPixels: 1e10
});
print("LST statistics (°C):", lstStats);

// CH4 stats
var ch4Stats = ch4Annual.reduceRegion({
reducer: ee.Reducer.mean().combine({
reducer2: ee.Reducer.stdDev(), sharedInputs: true
}).combine({
reducer2: ee.Reducer.minMax(), sharedInputs: true
}),
geometry: shallowWaterMask.geometry(),
scale: 30,
maxPixels: 1e10
});
print("CH4 annual flux (g/m²/yr):", ch4Stats);

// Total
var areaStats = shallowWaterMask.reduceRegion({
reducer: ee.Reducer.sum(),
geometry: roi,
scale: 10,
maxPixels: 1e10
});
var totalArea_km2 = ee.Number(areaStats.get("shallow_mask")).multiply(100).divide(1e6);

var totalCH4 = ch4Annual.reduceRegion({
reducer: ee.Reducer.sum(),
geometry: shallowWaterMask.geometry(),
scale: 30,
maxPixels: 1e10
});
var totalCH4_t = ee.Number(totalCH4.get("CH4_g_m2_yr")).multiply(900).divide(1e6); // 30m pixel

print("\nAREA:", totalArea_km2, "km²");
print("TOTAL CH4:", totalCH4_t, "t/yr");
print("CO2-eq:", totalCH4_t.multiply(GWP_CH4).divide(1000), "kt/yr");

// Expected values
print("\n═══════════════════════════════════════════════════════");
print("EXPECTED (from Python, 200 days):");
print("  CH4 moment mean: ~0.047 µmol/m²/s");
print("  Annual: ~11.5 g/m²/yr");
print("  Total (440 km²): ~5040 t/yr");
print("  CO2-eq: ~141 kt/yr");
print("═══════════════════════════════════════════════════════");

//——————————————————
// 7. VISUALIZATION
//——————————————————

Map.centerObject(roi, 10);

Map.addLayer(lstMean, {
min: 10, max: 25,
palette: ["blue", "cyan", "yellow", "orange", "red"]
}, "LST Mean (°C)");

Map.addLayer(ch4Annual, {
min: 0, max: 10,
palette: ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]
}, "CH4 Annual (g/m²/yr)");

Map.addLayer(shallowWaterMask.selfMask(), {palette: ["cyan"], opacity: 0.3}, "Shallow Mask");

//——————————————————
// 8. EXPORT
//——————————————————

var EXPORT_FOLDER = "Carbon_CH4_Final";
var EXPORT_CRS = "EPSG:32639";

// Export CH4_moment (for MC)
Export.image.toDrive({
image: ch4MomentMean.float(),
description: "CH4_Moment_Mean_umol_s",
folder: EXPORT_FOLDER,
region: roi,
scale: 30,
crs: EXPORT_CRS,
maxPixels: 1e12,
formatOptions: {cloudOptimized: true}
});

// Export LST_mean (for MC)
Export.image.toDrive({
image: lstMean.float(),
description: "LST_Mean_Season_C",
folder: EXPORT_FOLDER,
region: roi,
scale: 30,
crs: EXPORT_CRS,
maxPixels: 1e12,
formatOptions: {cloudOptimized: true}
});

// Export annual fluxes
Export.image.toDrive({
image: ee.Image.cat([ch4Annual, ch4CO2eq]).float(),
description: "CH4_Annual_Fluxes",
folder: EXPORT_FOLDER,
region: roi,
scale: 30,
crs: EXPORT_CRS,
maxPixels: 1e12,
formatOptions: {cloudOptimized: true}
});

// Export mask
Export.image.toDrive({
image: shallowWaterMask.unmask(0).toByte(),
description: "Shallow_Water_Mask_CH4",
folder: EXPORT_FOLDER,
region: roi,
scale: 10,
crs: EXPORT_CRS,
maxPixels: 1e12,
formatOptions: {cloudOptimized: true}
});

// Export observation count
Export.image.toDrive({
image: obsCount.int16(),
description: "CH4_Observation_Count",
folder: EXPORT_FOLDER,
region: roi,
scale: 30,
crs: EXPORT_CRS,
maxPixels: 1e12,
formatOptions: {cloudOptimized: true}
});

print("\n✅ Ready for Export");

/*

- ════════════════════════════════════════════════════════════════════════════
- SUMMARY FOR MC ANALYSIS
- ════════════════════════════════════════════════════════════════════════════
- 
- Input for MC: CH4_Moment_Mean_umol_s.tif
- - This is CH4 at satellite overpass (~10:30)
- - Units: µmol m⁻² s⁻¹
- 
- MC Parameters:
- - RMSE = 0.026 µmol/m²/s (add per-pixel noise)
- - F_CH4 = 0.876 ± 0.10 (moment → daily)
- - Ice-free days = 200 ± 10 (April 20 - November 5)
- - Conversion: 1.3858 (µmol/s → g/day)
- 
- Formula in MC:
- CH4_annual = CH4_moment × F_CH4 × 1.3858 × ice_free_days
- 
- EXPECTED RESULTS (200 days):
- Annual: ~11.5 g CH₄ m⁻² yr⁻¹
- Total (440 km²): ~5040 t CH₄ yr⁻¹
- CO₂-eq: ~141 kt CO₂-eq yr⁻¹
- 
- ════════════════════════════════════════════════════════════════════════════
  */
