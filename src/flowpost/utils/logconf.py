import logging
import logging.config

FORMAT = "%(message)s"
FORMAT = "%(funcName)s:%(lineno)d " "%(message)s"  #  What to add in the message
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "main": {
            "format": "%(levelname)s:%(module)s:%(funcName)s:%(lineno)d " "%(message)s",
            "datefmt": "[%X]",
            }
        },
    "handlers": {
        "console": {
            "formatter": "main",
            'class':'logging.StreamHandler',
            "level": "DEBUG",
        }
    },
    "loggers": {"": {"handlers": ["console"],
    'propagate': True}},
}

logging.config.dictConfig(LOGGING)

