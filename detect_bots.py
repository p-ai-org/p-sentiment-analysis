import botometer
import twitter_auth
keys = twitter_auth.get_keys()

cap_type = "english"  # "english" or "universal"
bot_threshold = 0.6
bom = None

def main():
    create_bom()
    print(is_bot("@barackobama"))
    # print(is_bot("@autonomices"))
    # print(is_bot("@fttechnews"))

def create_bom():
    global bom
    bom = botometer.Botometer(wait_on_ratelimit=True, consumer_key=keys["consumer_key"], consumer_secret=keys["consumer_secret"], access_token=keys["access_token"], access_token_secret=keys["access_token_secret"], rapidapi_key=keys["rapidapi_key"])
    # Next step: test with botometer_api_url="https://botometer-pro.p.rapidapi.com"

def bot_probability(account_id):
    return bom.check_account(account_id)["cap"][cap_type]

def is_bot(account_id):
    return bot_probability(account_id) > bot_threshold

if __name__ == "__main__":
    main()