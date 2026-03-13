"""
data_pipeline
=============
Handles satellite and climate data ingestion from:
  - NASA EarthData (NSIDC sea ice, MODIS land surface temperature)
  - Google Earth Engine (Sentinel, Landsat, ERA5 reanalysis)
  - NOAA (sea surface temperature, climate indices)

Entry point: data_pipeline.fetcher.DataFetcher
"""
