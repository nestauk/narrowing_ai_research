from altair_saver import save
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv, find_dotenv
import os

import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir

fig_path = f"{project_dir}/reports/figures"

# Checks if the right paths exist and if not creates them when imported
if os.path.exists(f"{fig_path}/png") == False:
    os.mkdir(f"{fig_path}/png")

if os.path.exists(f"{fig_path}/html") == False:
    os.mkdir(f"{fig_path}/html")


def altair_visualisation_setup():
    # Set up the driver to save figures as png
    load_dotenv(find_dotenv())
    # driv = webdriver.Chrome(os.getenv('chrome_driver_path'))
    driver = webdriver.Chrome(ChromeDriverManager().install())

    return driver


def save_altair(fig, name, driver, path=fig_path):
    """Saves an altair figure as png and html"""
    print(path)
    save(
        fig,
        f"{path}/png/{name}.png",
        method="selenium",
        webdriver=driver,
        scale_factor=5,
    )
    fig.save(f"{path}/html/{name}.html")


if __name__ == "__main__":
    altair_visualisation_setup()
