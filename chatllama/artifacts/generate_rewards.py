import argparse
import json

from langchain import OpenAI, LLMChain, PromptTemplate


class ScoreGenerator:
    def __init__(self) -> None:


        # initialize LLM and LangChain
        # openai_llm = OpenAI(
        #     model_name=llm_model,
        #     temperature=llm_temperature,
        #     max_tokens=llm_max_tokens,
        # )

        # Customaize your own Reward template by changing the
        # prompt_template
        print("init called")

    def distill(
        self,
        dataset_path: str,
    ) -> None:
        
    # [input (from QA dataset), response (internal llm), score (heuristic)]

        """Parse the dataset and assign scores using LLMs
        then save back the dataset with the uploaded scores
        """

        print("Assigning scores to the reward dataset...")

        # load the dataset
        with open(dataset_path, "r") as f:
            train_data = json.load(f)

        # for each element of the dataset, assign a score.
        for i, data in enumerate(train_data):
            if data.get("score", None) is None:

                user_input = data["user_input"]
                completion = data["completion"]
                print(
                    f"#### Data {i}:\n"
                    f"#### User_input:\n {user_input}\n"
                    f"#### Completion:\n {completion}\n"
                )
                
                assert len(data["completions"]) > 0
                score = float(len(data["completion"].split()))

                data["score"] = score
                print(f"### Score: {score} \n\n")

        # remove all the data that have no score
        train_data = [data for data in train_data if data.get("score", None)]
        # save the dataset back
        print("Writing the updated dataset back to disk ... ")
        with open(dataset_path, "w") as f:
            json.dump(train_data, f)

        print("Score Assignment Completed")

if __name__ == "__main__":

    # REWARD_TEMPLATE = dict(
    #     template=(
    #         "You have to evaluate the following chat with a score"
    #         "between 0 and 5"
    #         "You MUST evaluate: text quality, content quality and"
    #         "coherence.\n"
    #         "You MUST return only the number that represents your"
    #         "judgment.\n"
    #         "The input of the user is: {user_input}\n"
    #         "The output of the chatbot is: {completion}\n"
    #         "The score is:\n"
    #     ),
    #     input_variables=["user_input", "completion"],
    # )

    # Setup argument parser
    parser = argparse.ArgumentParser(
        prog="generate_rewards.py",
        description="Generate rewards using LangChain and LLMs",
    )

    parser.add_argument("dataset_path", help="Path to the dataset")
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     help="Specify the model to be used",
    #     default="text-davinci-003",
    # )
    # parser.add_argument(
    #     "-t",
    #     "--temperature",
    #     help="Specify the temperature of the score assignment",
    #     default=0.5,
    # )
    # parser.add_argument(
    #     "-k",
    #     "--max_tokens",
    #     help="Specify the max tokens of the score assignement",
    #     default=2048,
    # )

    # parser.add_argument(
    #     "-r",
    #     "--reward_template",
    #     help="Specify the reward template to be used",
    #     default=None,
    # )

    # parse arguments
    args = parser.parse_args()

    # if args.reward_template:
    #     templates = json.loads(args.reward_template)
    #     if templates.get("reward", None) is None:
    #         rw_template = REWARD_TEMPLATE
    #     else:
    #         rw_template = templates["reward"]
    # else:
    #     rw_template = REWARD_TEMPLATE

    score_generator = ScoreGenerator()

    score_generator.distill(args.dataset_path)
