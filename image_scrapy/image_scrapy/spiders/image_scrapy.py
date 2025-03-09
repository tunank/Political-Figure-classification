import scrapy
from image_scrapy.items import ImageScrapyItem


class ImageScrapySpider(scrapy.Spider):
    name = 'image_scrapy'
    search_term = "elon-musk-portrait"  # Fixed typo in "portrait"
    page_number = 2

    # customize setting to restrict the number of images to download
    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 80,
        'CLOSESPIDER_PAGECOUNT': 10  # Added page limit as a backup
    }

    def start_requests(self):
        url = f"https://www.gettyimages.ca/photos/{self.search_term}?page={self.page_number}"
        yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):
        all_images = response.xpath('//picture/img/@src').getall()

        item = ImageScrapyItem()
        item['image_urls'] = all_images
        yield item

        self.page_number += 1
        next_page = f"https://www.gettyimages.ca/photos/{self.search_term}?page={self.page_number}"
        yield response.follow(url=next_page, callback=self.parse)