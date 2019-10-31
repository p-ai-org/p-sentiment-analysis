import tweepy
 
auth =tweepy.OAuthHandler('BWDvPWSmrJ18xEBTPbTkxiGm9', 'FodmJZP8RPjdG5ZJ0dz6xpNEjOFYG5LnFTjvmKKze8e0GmAO8b')
auth.set_access_token('769205140645642240-pFoG4e2EpEQft63BjruaLmLvuehRQDx', 'NvpPKD8xZKd14NxmuzONg2rAApMjYkJK5wmkyE1UGgBOk')
 
api =tweepy.API(auth)
 
public_tweets =api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)