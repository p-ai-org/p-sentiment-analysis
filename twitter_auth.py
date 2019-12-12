def get_keys(path="../twitter_keys.txt"):
    consumer_key, consumer_secret, access_token, access_token_secret = open("../twitter_keys.txt", "r").read().splitlines()
    return {"consumer_key": consumer_key, "consumer_secret": consumer_secret, "access_token": access_token, "access_token_secret": access_token_secret}