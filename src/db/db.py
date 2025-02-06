from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base

# Database setup
DATABASE_NAME = "database.db"
database_url = f"sqlite:///{DATABASE_NAME}"
engine = create_engine(database_url, connect_args={"check_same_thread": False})
Base = declarative_base()
