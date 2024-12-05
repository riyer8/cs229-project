import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException

def init_browser():
    options = Options()
    options.headless = False  # Switch to True after testing
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    options.set_preference("general.useragent.override", user_agent)
    return webdriver.Firefox(options=options, service=Service())

def random_delay():
    time.sleep(random.uniform(2, 5))

def simulate_mouse_movement(driver):
    actions = ActionChains(driver)
    for _ in range(random.randint(10, 20)):
        actions.move_by_offset(random.randint(-10, 10), random.randint(-10, 10)).perform()
        random_delay()

def get_reuters_titles(ticker: str, max_titles: int = 10):
    base_url = f"https://www.reuters.com/search/news?query={ticker}"
    titles = []

    driver = init_browser()

    try:
        driver.get(base_url)
        print("Please complete CAPTCHA if prompted...")
        time.sleep(10)  # Allow page to load or complete CAPTCHA
        simulate_mouse_movement(driver)
        scroll_to_bottom(driver)

        articles = driver.find_elements(By.CSS_SELECTOR, "div.search-results__list div.story-content a")
        
        for article in articles[:max_titles]:
            try:
                title_text = article.text.strip()
                title_link = article.get_attribute("href")
                if title_text:
                    titles.append({"title": title_text, "link": title_link})
            except Exception as e:
                print(f"Error fetching an article: {e}")

    except TimeoutException:
        print("Error: Page took too long to load.")
    finally:
        driver.quit()

    return titles

def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        random_delay()
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            scroll_attempts += 1
            if scroll_attempts > 2:
                break
        else:
            last_height = new_height
            scroll_attempts = 0

if __name__ == "__main__":
    ticker = input("Enter the stock ticker (e.g., MSFT): ").strip()
    max_titles = int(input("Enter the maximum number of titles to fetch: "))

    results = get_reuters_titles(ticker, max_titles)

    if results:
        print("\nTop News Titles from Reuters:")
        for i, article in enumerate(results, start=1):
            print(f"{i}. {article['title']} - {article['link']}")
        pd.DataFrame(results).to_csv(f"{ticker}_reuters_news.csv", index=False)
        print(f"\nSaved results to {ticker}_reuters_news.csv")
    else:
        print("\nNo articles were found.")