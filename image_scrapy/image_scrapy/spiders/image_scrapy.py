import scrapy
from image_scrapy.items import ImageScrapyItem

class ImageScrapySpider(scrapy.Spider):
    name = 'image_scrapy'
    start_urls = ['https://www.gettyimages.ca/photos/joe-biden']

    # customize setting to restrict the number of images to download
    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 30
    }

    def parse(self, response, **kwargs):
        all_images = response.xpath('//picture/img/@src').getall()

        item = ImageScrapyItem()
        item['image_urls'] = all_images
        yield item