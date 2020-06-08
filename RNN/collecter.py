from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from fredapi import Fred
import yfinance

import re
import time
import csv
import os


class DataCollecter(object):
    def __init__(self, url, filepath):
        self.url = url
        self.filepath = filepath
        self.api_key = 'e442a32f70d944000ad2448d804b9776'
        self.fred = Fred(api_key=self.api_key)
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding = 'utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["date", "time", "price"])

    def current_data(self):
        delay = 3
        browser = webdriver.Chrome()
        browser.get(self.url)
        WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.ID, "quote-header-info")))
        data = browser.page_source
        browser.quit()
        soup = BeautifulSoup(data, "html.parser")

        try:
            price = soup.find_all("span", class_="Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)")[0]
            value = float(re.findall(r'>[0-9,.]+<', str(price))[0].replace('>', '').replace('<', '').replace(',', ''))
            day = time.strftime("%Y-%m-%d", time.localtime())
            t = time.strftime("%H:%M:%S", time.localtime())

            with open(self.filepath, 'a+', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([day, t, value])
            
            return [day, t, value]

        except:
            print("Cannot Get Any Data.")

    def load_data_from_fred(self, series):
        data = self.fred.get_series(series)
        #print(data)

        return data

    def load_history_from_yahoo(self, target):
        data = yfinance.Ticker(target)
        history = data.history(period = 'max')
        #print(history)

        return history