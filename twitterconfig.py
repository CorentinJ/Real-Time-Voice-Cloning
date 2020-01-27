import tweepy
import logging
import os

logger = logging.getLogger()

def create_api():
    """
    Use your secret twitter keys here instead. These are fake
    """
    consumer_key = "mB99fjW7s4vL61JYe" # os.getenv("CONSUMER_KEY")
    consumer_secret = "5cm6xk8VC2ZkMi28HioS3oletFEEzJGUbGw" # os.getenv("CONSUMER_SECRET")
    access_token = "236084SuIM4ujFN8F" # os.getenv("ACCESS_TOKEN")
    access_token_secret = "YZNJHPUbOhfJ5KkBswYfriP6s6KHU" # os.getenv("ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, 
        wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    logger.info("API created")
    return api
