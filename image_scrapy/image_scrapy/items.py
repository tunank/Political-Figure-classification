# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ImageScrapyItem(scrapy.Item):
    image_urls = scrapy.Field()  # List of image URLs to download
    images = scrapy.Field()  # Stores metadata about downloaded images
