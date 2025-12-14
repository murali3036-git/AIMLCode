
from abc import ABC , abstractmethod
class MathsBase(ABC):
    def __init__(self, Num1 , Num2):
      self.Num1 = Num1
      self.Num2 = Num2
    
    @abstractmethod
    def Add(self):
        pass

class MyMaths(MathsBase):
     def Add(self):
        return self.Num1+self.Num1

class AdvanceMaths(MyMaths):
   def Log(self):
      print( " this logg ")
   @property
   def Num1(self):
        return self._num1
   
   @Num1.setter
   def Num1(self,value):
        if(value<0):
            raise ValueError("Negative not allowed")
            self._num1 = value
   
   @property
   def Num2(self):
        return self._num2
   @Num2.setter
   def Num2(self,value):
         if(value<0):
             raise ValueError("Negative not allowed")
             self._num2 = value
