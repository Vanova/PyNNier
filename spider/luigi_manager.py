import random
import luigi
from luigi.mock import MockFile

# 1. crawl the image links
# 2. parse proxies
# 3. download images with current proxies
# 4. filter and reshape images
# 5. fine-tune network

class SimpleTask(luigi.Task):
    """
    This simple task prints Hello World!
    """

    def output(self):
        return MockFile("SimpleTask", mirror_on_stderr=True)

    def run(self):
        _out = self.output().open('w')
        _out.write(u"Hello World!\n")
        _out.close()


def parse_image_urls(url):
    pass


def download_image(url):
    name = random.randrange

def preprocess_image():
    pass


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=SimpleTask)