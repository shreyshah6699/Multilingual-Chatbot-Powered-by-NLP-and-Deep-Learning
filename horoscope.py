from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

def get_horoscope(sign):
    # sign = input("Enter Sun Sign: ")
    
    # Path to the uBlock Origin .crx file
    ublock_path = 'uBlock-Origin.crx'

    chrome_options = Options()
    chrome_options.add_extension(ublock_path)
    chrome_options.set_capability("pageLoadStrategy", "eager")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(f"https://www.washingtonpost.com/entertainment/horoscopes/{sign.lower()}/")
    
    time.sleep(3)  # Wait for the page to load

    class_name = "mt-sm.mb-sm.font-sm.gray-dark"
    horoscope_element = driver.find_element(By.CLASS_NAME, class_name)
    horoscope_text = horoscope_element.text
    
    return horoscope_text