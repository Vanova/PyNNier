import logging




def get_logger(file_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(file_name)

    return logger


def get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    formatted_time = '%d:%02d:%02d' % (h, m, s)

    return formatted_time
