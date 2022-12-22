import os
import json
import openai
import cohere
import requests
from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import LLMMathChain, SerpAPIWrapper, ConversationChain
from langchain.agents import initialize_agent, Tool



class assistant:

    def __init__(self, ph_id : str, ultramsg_token : str, ultramsg_instance : str):
        """
        Initialize a new instance of the `assistant` class.
        
        Args:
            ph_id (str): The ID of the user.
            ultramsg_token (str): The UltraMsg API token.
            ultramsg_instance (str): The UltraMsg instance name.
        """
        self.ph_id = ph_id
        self.ultramsg_token = ultramsg_token
        self.ultramsg_instance = ultramsg_instance
        
        # Classification categories
        self.classifications = ("Logical", 
                                "Mathematical", 
                                "Philosophical", 
                                "Emotional", 
                                "Poetic", 
                                "Search"
                                )
        
        # Emotion categories
        self.emotions = ("happiness",
                        "sadness", 
                        "anger", 
                        "disgust", 
                        "hate", 
                        "regret", 
                        "love", 
                        "fear", 
                        "surprise", 
                        "anxiety", 
                        "satisfaction", 
                        "amusement",
                        "shame",
                        "guilt",
                        "pride", 
                        "embarrassment",
                        "contempt", 
                        "boredom",
                        "intrest",
                        "awe",
                        "envy",
                        "admiration",
                        "nostalgia",
                        "hatred",
                        "calmness",
                        "relief",
                        "compassion",
                        "neutral"
                        )



    def authenticate(self, openai_key : str, cohere_key : str, serpapi_key : str):
        """
        Authenticate the API keys for OpenAI, Cohere, and SerpAPI.
        
        Args:
            openai_key (str): The API key for OpenAI.
            cohere_key (str): The API key for Cohere.
            serpapi_key (str): The API key for SerpAPI.
        """
        openai.api_key = openai_key
        self.co = cohere.Client(api_key=cohere_key)



    def request_process(self, data : json):
        """
        Process a request for text processing.
        
        Args:
            data (json): The request data in JSON format.
        
        Returns:
            tuple: A tuple containing a boolean indicating whether the request should be processed, and the processed data.
        """
        data = self.text_processing(data)
        if( data["sender id"] != self.ph_id and data["sender id"] != ''):# and data["message"][0] == '/' ):
            return True, data
        else:
            return False, data




    def classifiers(self, message : str): 

        prompt = 'From a list of classifications '+str(self.classifications)+' select best-suited classification for the sentences given below.\n'
        prompt = prompt +'"If A is greater than B and B is greater than C, then A must be greater than C." (Logical)\n'
        prompt = prompt +'"What is the sum of the first five whole numbers?" (Mathematical)\n'
        prompt = prompt +'"Is there such a thing as absolute truth?" (Philosophical)\n'
        prompt = prompt +'"Write a python code to demonstrate the use of async functions." (Logical)\n'
        prompt = prompt +'"How do you feel about your current situation?" (Emotional)\n'
        prompt = prompt +'"What is the beauty of a star-filled night?" (Poetic)\n'
        prompt = prompt +'"What is the latest news on the US-China trade war?" (Search)\n'
        prompt = prompt +'"If X is an even number, then X+2 must also be an even number." (Logical)\n'
        prompt = prompt +'"What is the equation for the area of a triangle?" (Mathematical)\n'
        prompt = prompt +'"What is the meaning of life?" (Philosophical)\n'
        prompt = prompt +'"How do you feel when you are surrounded by love?" (Emotional)\n'
        prompt = prompt +'"How do I deal with my feelings of anger and frustration?" (Emotional)\n'
        prompt = prompt +'"Is it logically possible for a triangle to have four sides?" (Logical)\n'
        prompt = prompt +'"What is the mathematical formula for calculating the area of a circle?" (Mathematical)\n'
        prompt = prompt +'"What is the meaning of life?" (Philosophical)\n'
        prompt = prompt +'"How do I deal with my feelings of anger and frustration?" (Emotional)\n'
        prompt = prompt +'"What is the significance of the daffodils in the poem \'I Wandered Lonely as a Cloud\' by William Wordsworth?" (Poetic)\n'
        prompt = prompt +'"What is the current state of relations between the US and North Korea?" (Search)\n'
        prompt = prompt +'"What are javascript async functions, literals and difference between python and c++" (Logical)\n'
        prompt = prompt +'"Is it possible to prove the existence of God through reason alone?" (Philosophical)\n'
        prompt = prompt +'"What is the square root of 64?" (Mathematical)\n'
        prompt = prompt +'"What is the main argument in Plato\'s \'The Republic\'?" (Philosophical)\n'
        prompt = prompt +'"How does the rhyme scheme in the poem \'The Road Not Taken\' by Robert Frost contribute to its overall meaning?" (Poetic)\n'
        prompt = prompt +'"What is the meaning of life?" (Philosophical)\n'
        prompt = prompt +'"What is the square root of 64?" (Mathematical)\n'
        prompt = prompt +'"What is the price of bitcoin?" (Search)\n'
        prompt = prompt +'"'+message+'"'
        

        response = self.co.generate(
            model='xlarge',
            prompt=prompt,
            max_tokens=5,
            temperature=0.3,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=[],
            return_likelihoods='NONE')
        response = response.generations[0].text


        start = response.find('(')

        # Find the position of the closing parenthesis
        end = response.find(')')

        # Extract the substring between the opening and closing parentheses
        response = response[start+1:end]

        return response



    def emotion_processing(self, message : str, emotion : str):

        prompt = 'From a list of emotions '+str(self.emotions)+' select best-suited emotion for the sentences given below.'
        prompt = prompt +'\n"I am so happy to see you!" (happiness)'
        prompt = prompt +'\n"I am feeling so sad today." (sadness)'
        prompt = prompt +'\n"I am so angry right now!" (anger)'
        prompt = prompt +'\n"I feel disgusted by what you just said." (disgust)'
        prompt = prompt +'\n"I hate you." (hate)'
        prompt = prompt +'\n"I regret not studying for that exam." (regret)'
        prompt = prompt +'\n"I love you more than anything." (love)'
        prompt = prompt +'\n"I am afraid of the dark." (fear)'
        prompt = prompt +'\n"I am surprised by that news." (surprise)'
        prompt = prompt +'\n"I am anxious about my upcoming job interview." (anxiety)'
        prompt = prompt +'\n"I am satisfied with my performance on that project." (satisfaction)'
        prompt = prompt +'\n"That joke was so funny, I am amused." (amusement)'
        prompt = prompt +'\n"I feel ashamed of what I did." (shame)'
        prompt = prompt +'\n"I feel guilty for not telling the truth." (guilt)'
        prompt = prompt +'\n"I am proud of my accomplishments." (pride)'
        prompt = prompt +'\n"I am embarrassed by my mistake." (embarrassment)'
        prompt = prompt +'\n"I feel contempt towards that person." (contempt)'
        prompt = prompt +'\n"I am bored by this conversation." (boredom)'
        prompt = prompt +'\n"I am interested in learning more about that topic." (interest)'
        prompt = prompt +'\n"I feel awe at the beauty of nature." (awe)'
        prompt = prompt +'\n"I envy her success." (envy)'
        prompt = prompt +'\n"I admire his bravery." (admiration)'
        prompt = prompt +'\n"I have a feeling of nostalgia for my childhood." (nostalgia)'
        prompt = prompt +'\n"I feel hatred towards that person." (hatred)'
        prompt = prompt +'\n"I am feeling calm and relaxed." (calmness)'
        prompt = prompt +'\n"I am relieved that it\'s over." (relief)'
        prompt = prompt +'\n"'+message+'"'


        response = self.co.generate(
            model='xlarge',
            prompt=prompt,
            max_tokens=5,
            temperature=0.3,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=[],
            return_likelihoods='NONE')
        response = response.generations[0].text


        start = response.find('(')

        # Find the position of the closing parenthesis
        end = response.find(')')

        # Extract the substring between the opening and closing parentheses
        response = response[start+1:end]

        return response



    def langchain_chaining(self, message):
        # Agent
        # Load the tool configs that are needed 
        llm = OpenAI(temperature=0)
        search = SerpAPIWrapper()
        llm_math_chain = LLMMathChain(llm=llm, verbose=True)
        conversation_chain = ConversationChain(llm=llm)
        tools = [
            Tool(
                name = "Search",
                func = search.run,
                description = "useful for when you need to answer questions about current events"
            ),
            Tool(
                name = "Calculator",
                func = llm_math_chain.run,
                description = "useful for when you need to answer questions about math"
            ),
            Tool(
                name = "Conversation",
                func = conversation_chain.run,
                description = "useful when in conversation"
            )
        ]

        # Construct the agent 
        #llm = Cohere(temperature=0)
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

        agent_response = agent.run(message)

        return agent_response



    def general_prompt(self, message : str, temperature = 0.5, emotion = "interest"):
        try:

            prompt="Techie is a person with knowledge in the field of science, maths, philosophy, history, geography and everything."
            prompt = prompt +" Techie is also funny hilarious and smart while talking. He also likes casual talks. Techie's best friend is aswanth. Also he is the one who created Techie. Aswanth is born in 13-12-2000 and Techie is born in same data"
            prompt = prompt +"User is in conversation with Techie, Techie is responding with emotion \""+emotion+"\".\nUser: Hi\nTechie: Hello, how are you?\nUser: "+message

            if prompt[-1] != '.' or prompt[-1] != '?':
                prompt = prompt + '.\nTechie: '
            else:
                prompt = prompt + '\nTechie: '

            response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=temperature, max_tokens=(4000-len(prompt)))
            response_text = str(response['choices'][0]['text'])
            print(response_text)
            return(response_text)

        except Exception as e:
            print(e,"in general_prompt")



    def readmessage(self, chatid : str):
        try:
            url = "https://api.ultramsg.com/instance"+self.ultramsg_instance+"/chats/messages"

            querystring = {"token":self.ultramsg_token,"chatId":chatid,"limit":"1"}
            headers = {'content-type': 'application/x-www-form-urlencoded'}

            response = requests.request("GET", url, headers=headers, params=querystring).json()
            response = response[0]["body"]

            return response

        except Exception as e:
            print(e,"in readmessage")



    def text_processing(self, data : json):
        try:
            message = data['data']

            if message != []:
                msg_from = message['from']
                msg_text = message['body']

                return {"sender id": msg_from, "message": msg_text}

            else:    

                return {"sender id": msg_from[0], "message": ''}

        except Exception as e:
            print(e,"in text_processing")



    def reply(self, to : str, body : str):
        
        try:
            to = '+'+to.split('@')[0]
            body = body.replace("’", "'")

            url = "https://api.ultramsg.com/instance"+self.ultramsg_instance+"/chats/messages"

            payload = "token="+self.ultramsg_token+"="+to+"&body="+body+"&priority=10&referenceId="
            headers = {'content-type': 'application/x-www-form-urlencoded'}

            response = requests.request("POST", url, data=payload, headers=headers)

        except Exception as e:
            print(e,"in reply")



    def process(self, sender_id : str, message : str):

        if message[0]=='/':
            message = message.split('/')[1]
        redirect = False
        emotion = "neutral"
        temperature = 0.5
        generalclassification = self.classifiers(message)

        if generalclassification == "Emotional":
            self.emotion_processing(message, generalclassification)
            emotion = "compassion"
            temperature=0.8

        elif generalclassification == "Poetic":
            emotion = "satisfaction"
            temperature=0.9

        elif generalclassification == "Philosophical":
            emotion = "satisfaction and calmness"
            temperature=0.9

        elif generalclassification == "Logical":
            emotion = "interest"
            temperature=0.1

        elif generalclassification == "Search":
            redirect = True
            ai_response = self.langchain_chaining(message)

        elif generalclassification == "Mathematical":
            redirect = True
            try:
                ai_response = self.langchain_chaining(message)
            except:
                redirect = False
        
        if not redirect:
            ai_response = self.general_prompt(message, temperature, emotion)

        print(ai_response)

        self.reply(sender_id, ai_response)