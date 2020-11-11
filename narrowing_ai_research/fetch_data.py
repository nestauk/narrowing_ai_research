import logging
from narrowing_ai_research.data.webscraping import webscraping
from narrowing_ai_research.data.fetch_figshare import fetch_figshare

logging.info("fetching figshare data")
fetch_figshare()

logging.info("scraping deepmind and open AI papers")
webscraping()
