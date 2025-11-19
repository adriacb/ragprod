from abc import ABC, abstractmethod

class BaseChunk(ABC):

    @abstractmethod
    def get_docid():
        pass
    
