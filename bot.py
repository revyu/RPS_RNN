from numpy.random import choice
import typing
from typing import Literal

def get_table_decide_winner_classical(modulus):
    table = [["l" for i in range(modulus)] for i in range(modulus)]

    for row in range(modulus):
        table[row][row] = "t"  # диагонали дают ничью
        for column in [(row+i) % modulus for i in range(1, (modulus-1)//2+1)]:
            table[row][column] = "w"  
    
    return table



class bot():
    def __init__(self, number_of_states:int ,strategy:Literal["mrugesh","kris","mrugesh_r","kris_r"],
                 mrugesh_n=10, classic_mode=True, initial_history=None):
        #number_of_states -количество состояний для игры , =3 дает классические правила
        #strategy - стратегия которой придерживается бот


        #to_do: реализовать произвольные правила
        self.number_of_states=number_of_states
        self.strategy=strategy
        if self.strategy=="kris" or self.strategy=="kris_r" :
            if initial_history==None:
                self.bot_history=[0]
            else:
                self.bot_history=initial_history
        if self.strategy=="mrugesh" or self.strategy=="mrugesh_r":
            if initial_history==None:
                self.bot_history=[0 for i in range(mrugesh_n)]
            else:
                self.bot_history=initial_history
            self.mrugesh_n=mrugesh_n
        if classic_mode:
            self.decide_winner=get_table_decide_winner_classical(number_of_states)
        
    

    def update_history(self,move):
        self.bot_history.append(move)
    
    # возвращает первый из выигрышных вариантов на самый частый ход из n последних 
    def _mrugesh(self,n):
        last_ten=self.bot_history[-n:]
        most_frequent = max(set(last_ten), key=last_ten.count)

        ideal_response = (most_frequent+1)%self.number_of_states
        return ideal_response

    # возвращает случайный выигрышный вариант на самый частый ход из n последних
    #- должен быть согласован с play
    def _mrugesh_r(self,n):
        last_ten=self.bot_history[-n:]
        most_frequent = max(set(last_ten), key=last_ten.count)

        
        return choice([(most_frequent+i) % self.number_of_states for i in range(1, (self.number_of_states-1)//2+1)])
    
    # возвращает первый из выигрышных вариантов на последний ход 
    def _kris(self):
        return (self.bot_history[-1]+1)%self.number_of_states
    
    # возвращает случайный из выигрышных вариантов на последний ход
    def _kris_r(self):
        return choice([(self.bot_history[-1]+i) % self.number_of_states for i in range(1, (self.number_of_states-1)//2+1)])
    
    def play(self,context):
        s=[]
        if self.strategy=="mrugesh":
            for i in range(len(context)):
                s.append(self._mrugesh(self.mrugesh_n))
                self.update_history(context[i])
        
        if self.strategy=="mrugesh_r":
            for i in range(len(context)):
                s.append(self._mrugesh_r(self.mrugesh_n))
                self.update_history(context[i])
            
        elif self.strategy=="kris":
            for i in range(len(context)):
                s.append(self._kris())
                self.update_history(context[i])
            
        elif self.strategy=="quincy":
            for i in range(len(context)):
                s.append(self._quincy())
                self.update_history(context[i])

        elif self.strategy=="abbey":
            for i in range(len(context)):
                s.append(self._abbey())
                self.update_history(context[i])
        return s





