from abc import ABC, abstractmethod
class CustomerBase(ABC):
    def __init__(self, name , amount):
      self.name = name
      self._amount = amount
    
    @abstractmethod
    def Add(self):
      pass
class Customer(CustomerBase):
    def Add(self):
       print("Add called")
    def display(self):
      print(self.name)

    @property
    def amount(self):
      return self._amount
    
    @amount.setter
    def amount(self,value):
       if(value <0 ):
          raise ValueError("value can not be negative")
       self._amount = value

class SpecialCustomer(Customer):
   def someMethod(self):
      print(self.name)