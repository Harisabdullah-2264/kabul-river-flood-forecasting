// ── APPENDIX A: ERA5-Land Extraction — GEE Script ───────────────────────────
// Author: Haris Abdullah (GCV2470), GIK Institute

// 1. Load the HydroSHEDS Level-5 basin polygon for Kabul River (Nowshera)
var basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_5");
var pt = ee.Geometry.Point([71.974, 34.015]); // Nowshera gauge
var basin = basins.filterBounds(pt).first();
var basinGeom = basin.geometry().simplify(1000);

// 2. Load ERA5-Land daily aggregated dataset
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .filterDate("1990-01-01", "2022-12-31")
  .select(["total_precipitation_sum","temperature_2m","snow_depth_water_equivalent"]);

// 3. Compute basin-mean for each day
var daily = era5.map(function(img) {
  var date = img.date().format("YYYY-MM-dd");
  var stats = img.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: basinGeom,
    scale: 11132,
    maxPixels: 1e9
  });
  return ee.Feature(null, stats).set("date", date);
});

// 4. Apply unit conversions and export
var converted = daily.map(function(f) {
  return f
    .set("P_mm",  ee.Number(f.get("total_precipitation_sum")).multiply(1000))
    .set("T_C",   ee.Number(f.get("temperature_2m")).subtract(273.15))
    .set("snow_mm", ee.Number(f.get("snow_depth_water_equivalent")).multiply(1000));
});

Export.table.toDrive({
  collection: converted,
  description: "ERA5Land_KabulBasin_1990_2022",
  fileFormat: "CSV",
  selectors: ["date","P_mm","T_C","snow_mm"]
});
