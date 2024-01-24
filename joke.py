from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_joke():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://www.ajokeaday.com/")

    try:
        # Wait for the joke element to load
        joke_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "jd-body"))
        )
        
        # Extract the joke text
        joke_text = joke_element.text.strip()

        return joke_text
    finally:
        driver.quit()