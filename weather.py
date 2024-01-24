from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

def get_weather(citystate):
	# citystate = input("Enter City, State: ")
	# city, state = citystate.split(",")
	
	ublock_path = 'uBlock-Origin.crx'

	chrome_options = Options()
	chrome_options.add_extension(ublock_path)

	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
	driver.get("https://www.wunderground.com/")

	try:
		# Find the search box and enter the location
		search_box = WebDriverWait(driver, 10).until(
			EC.presence_of_element_located((By.NAME, "query"))
		)
		# search_box.send_keys(f"{city}, {state}")
		search_box.send_keys(citystate)
		time.sleep(1)
		search_box.send_keys(Keys.RETURN)

		temperature_container = driver.find_element(By.CLASS_NAME, "current-temp")
		temperature = temperature_container.find_element(By.CSS_SELECTOR, ".wu-value.wu-value-to").text.strip()
		return temperature
	finally:
		driver.quit()
