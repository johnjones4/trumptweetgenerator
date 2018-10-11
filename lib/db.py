from sqlalchemy import create_engine, Index, Column, Integer, String, Text, DateTime, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from os import getenv

Base = declarative_base()

def init_db():
  global session
  dburl = getenv("DB_URL")
  engine = create_engine(dburl, echo=False)
  Base.metadata.create_all(engine)
  Session = sessionmaker(bind=engine)
  session = Session()

class Tweet(Base):
  __tablename__ = "tweet"
  id = Column(Integer, primary_key=True)
  tweet_id = Column(BigInteger, nullable=False, unique=True)
  date = Column(DateTime)
  text = Column(String(300))
