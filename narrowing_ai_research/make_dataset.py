# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv

# Important to import the module
# This configures logging, file-paths, model config variables
import narrowing_ai_research
from narrowing_ai_research.transformers.arxiv_tokenise import arxiv_tokenise
from narrowing_ai_research.transformers.find_ai_papers import find_ai_papers
from narrowing_ai_research.transformers.create_topic_df import create_topic_df
from narrowing_ai_research.transformers.process_paper_data import process_paper_data
from narrowing_ai_research.estimators.train_word2vec import train_word2vec

logger = logging.getLogger(__name__)


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    # config = narrowing_ai_research.config
    arxiv_tokenise()
    train_word2vec()
    find_ai_papers()
    create_topic_df()
    process_paper_data()

    return


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = narrowing_ai_research.project_dir

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    try:
        msg = f"Making datasets..."
        logger.info(msg)
        main()
    except (Exception, KeyboardInterrupt) as e:
        logger.exception("make_dataset failed", stack_info=True)
        raise e
