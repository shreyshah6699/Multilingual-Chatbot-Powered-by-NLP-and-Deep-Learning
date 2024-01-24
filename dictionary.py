from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time

def get_meaning(word):
	# word = input("Enter word: ")
	
	# Path to the uBlock Origin .crx file
	ublock_path = 'uBlock-Origin.crx'
	
	chrome_options = Options()
	chrome_options.add_extension(ublock_path)
	chrome_options.set_capability("pageLoadStrategy", "eager")

	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
	driver.get("https://www.dictionary.com")

	try:
		# Find the search box and enter the location
		search_box = WebDriverWait(driver, 10).until(
			EC.presence_of_element_located((By.ID, "global-search"))
		)
		search_box.send_keys(word)
		time.sleep(2)
		search_box.send_keys(Keys.RETURN)
		
		ol_elements = driver.find_elements(By.CSS_SELECTOR, "div[data-type='word-definitions'] ol")
		ol_text = [ol.text for ol in ol_elements]
		
		return ol_text[0]
	except:
		return None
	finally:
		driver.quit()
