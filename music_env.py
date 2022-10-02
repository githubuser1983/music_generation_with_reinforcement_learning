#!/usr/bin/env python

"""
Music environment for gym openai.
"""

# Core Library
import logging.config
import math
import random
from typing import Any, Dict, List, Tuple

# Third party
#import cfg_load
import gym
import numpy as np
#import pkg_resources
from gym import spaces
import music_utils
import copy


#path = "config.yaml"  # always use slash in packages
#filepath = pkg_resources.resource_filename("gym_banana", path)
#config = cfg_load.load(filepath)
#logging.config.dictConfig(config["LOGGING"])


class MusicEnv(gym.Env):
    """
    Define a Music environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """
    def __vec_format(self,notes):
        return music_utils.getMeanVectorForNotes(notes)
        #return music_utils.getConcatVectorForNotes(notes)
    

    def __init__(self,file="for_elise_by_beethoven.musicxml",maxNotes = 4,max_wrong_notes = 2,help_factor = 0.5) -> None:
        self.__version__ = "0.1.0"


        self.max_wrong_notes = max_wrong_notes
        # General variables defining the environment
        self.file = file
        self.maxDim = music_utils.maxDim

        scores,instruments = music_utils.parse_file(file)
        self.scores = scores
        self.instruments = instruments
        from collections import deque
        self.score = (scores[1])
        self.maxNotes = maxNotes
        self.possibleKeysToPress = list(set([ self.score[i][1][0] for i in range(len(self.score))]))
        #self.possibleKeysToPress = music_utils.noteslist
        self.action_space = spaces.Discrete(len(self.possibleKeysToPress))

        self.done = False
        
        factor = 1
        self.dimObs = factor*(music_utils.P.shape[1]+music_utils.D.shape[1]+music_utils.V.shape[1]+music_utils.R.shape[1])
        # Observation is the remaining time
        low = np.array([[-np.Inf]*self.dimObs]).transpose()  # remaining_tries
        high = np.array([[np.Inf]*self.dimObs]).transpose()  # remaining_tries
        self.observation_space = spaces.Box(low,high,shape=(self.dimObs,1),dtype=np.float32)

        print("dimObs = ", self.dimObs)
        print("#Act = ", self.action_space.n)
        self.note_counter = 0
        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory: List[Any] = []
        self.init_notes =  [ n[1][0] for n in self.score[0:(0+self.maxNotes)]]   
        self.init_observation =   self.__vec_format(self.init_notes)  
        self.notes = [x for x in self.init_notes]
        self.cumulative_reward = 0
        self.step_counter = 0
        self.note_counter = len(self.notes)
        self.observation = copy.deepcopy(self.init_observation)
        self.setOfKeysPressed = set([])
        self.wrong_note_counter = 0 
        X = []
        print("computing vectors..")
        for i in range(len(self.possibleKeysToPress)):
            nn = self.possibleKeysToPress[i]
            vv = music_utils.getVectorForNote(nn)
            X.append(vv)
            
        self.XX = np.array(X)    
        print("print computing knn..")
        self.nbrs = music_utils.get_knn_model(self.XX) 
        print("..done")
        self.dd = dict([])
        self.is_composing = False
        self.random_init = False
        self.nr_of_notes_to_compose = 200
        self.help_factor = help_factor
        
        

    def chose_random_action(self):
        keyExpected = self.score[(self.note_counter+1)%len(self.score)][1][0]
        lastNotes = tuple(self.notes)
        if (lastNotes,keyExpected) in self.dd.keys():
            ids = self.dd[(lastNotes,keyExpected)]
        else:
            ids = self.compute_allowed_actions()
        return np.random.choice(ids)
        
    def compute_allowed_actions(self):
        keyExpected = self.score[(self.note_counter+1)%len(self.score)][1][0]
        lastNotes = tuple(self.notes)
        #expected_action = self.possibleKeysToPress.index(keyExpected)
        vecExpected = music_utils.getVectorForNote(keyExpected)
        lastVec = music_utils.getMeanVectorForNotes(lastNotes)
        noteReprVec = self.XX[music_utils.findBestMatches(self.nbrs,lastVec,n_neighbors=1)[0][1]]
        dist = np.linalg.norm(noteReprVec-vecExpected)
        #print(dist)
        dx = music_utils.findByRadius(self.nbrs,vecExpected,radius = dist*(1.0-self.help_factor)+0.01)
        #nx = len(dx1)//2
        #dx = music_utils.findBestMatches(self.nbrs,vecExpected,n_neighbors = nx)
        #print(dx)
        #dx = music_utils.findByRadius(self.nbrs,vecExpected,radius = dist/2.0+0.1)
        ids = [d[1] for d in dx]
        #print(ids)
        self.dd[(lastNotes,keyExpected)] = ids
        #ids.append(expected_action)
        return ids
    
    
    def step(self, action: int) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.done:
            raise RuntimeError("Episode is done")
        keyPressed = self.possibleKeysToPress[action]
        
        keyVector = music_utils.getVectorForNote(keyPressed)
        
        keyExpected = self.score[(self.note_counter+1)%len(self.score)][1][0]
        someKeysExpected = [ self.score[(self.note_counter+i)%len(self.score)][1][0] for i in [1]]
        #lastKey = self.notes[-1]
        #lastPitch = lastKey[0]
        #pitchPressed = keyPressed[0]
        #reward = np.exp(-music_utils.kernNote(keyPressed,lastKey))
        #self.setOfKeysPressed.add(keyPressed)
        lastNotes = tuple(self.notes)      

        if (lastNotes,keyExpected) in self.dd.keys():
            nrOfPossibleActions = len(self.dd[(lastNotes,keyExpected)])
        else:
            nrOfPossibleActions = len(self.compute_allowed_actions())      
            
        self.step_counter += 1
        
        old_obs = self.observation
        
        self.notes.pop(0)
        self.notes.append(keyPressed)
        #reward = np.exp(-np.linalg.norm(music_utils.getVectorForNote(keyExpected)-keyVector))*np.log(nrOfPossibleActions+1)
        if keyPressed in someKeysExpected:
            reward = np.min([ 2*np.sqrt(2.0),np.log(nrOfPossibleActions+1)])
            self.note_counter += 1
            self.wrong_note_counter = 0 
        else:
            self.wrong_note_counter += 1
            #reward = -reward
            reward = -1.0*np.linalg.norm(music_utils.getVectorForNote(keyExpected)-keyVector)
        
        
        #print(keyPressed,lastKey,reward,self.setOfKeysPressed)
        
        
        #self.notes =  [ n[1][0] for n in self.score[self.note_counter:(self.note_counter+self.maxNotes)]]   
        #self.notes.pop(0)
        #self.notes.append(keyPressed)
        #lamb = 0.1
        self.observation =   self.__vec_format(self.notes)
        #newvec = self.__vec_format(self.maxNotes*[keyPressed])
        
        #dist = lamb*np.linalg.norm(keyVector-old_obs) #music_utils.findBestMatch(self.nbrs,self.observation) # distance to nearest neighbor in given music piece
        #reward = dist
        #print(reward)
        #print(dist)

        self.cumulative_reward += reward
        
        
        
        if not self.is_composing and  self.wrong_note_counter>=self.max_wrong_notes:
            self.note_counter = len(self.notes)
            self.cumulative_reward = 0.0
            self.done = True
        elif self.is_composing and self.step_counter == self.nr_of_notes_to_compose:
            self.done = True
            
        #print(state,self.note_counter)
        state = self.observation, reward, self.done, False, {"keyPressed":keyPressed}
        return state


    def reset(self) -> List[int]:
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        from collections import deque
        self.wrong_note_counter = 0
        self.step_counter = 0
        self.cumulative_reward = 0.0
        self.note_counter = 0
        self.done = False
        if self.is_composing and self.random_init:
            i = np.inf
            while i + self.maxNotes >= len(self.score):
                i = np.random.choice(len(self.score))
                self.notes  = [ n[1][0] for n in self.score[i:(i+self.maxNotes)%len(self.score)]]   
                self.note_counter = i+self.maxNotes
        else:    
            self.notes = [x for x in self.init_notes]
            self.note_counter = len(self.notes)
        self.setOfKeysPressed = set([])
        self.observation = music_utils.getMeanVectorForNotes(self.notes) #copy.deepcopy(self.init_observation)
        self.step_counter = 0
        
        reward = 0
        return self.observation,reward, self.done, False, {}

   
    def render(self):
        return None

    def _get_state(self) -> List[int]:
        """Get the observation."""
        ob = self.observation
        return ob

