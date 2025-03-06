import pickle

class OpticalModel:
    """Class for saving and loading complete computation results.
    
    This class serves as a container to serialize/deserialize computation results.
    After loading, it provides direct access to both the star configuration
    and all particle properties, including their material properties (prtl.matrl).

    Attributes:
        star (Star): Input star object
        prtl (Particles): Particles object containing all computed properties, 
                         including material properties via prtl.matrl
    """
    def __init__(self, star=None, prtl=None):
        if star is None and prtl is None:
            raise ValueError("At least one of star or prtl must be provided.")

        self.star = star
        self.prtl = prtl
    
    def save(self, file_name):
        """Save the complete model state to a file using pickle."""
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        
    @staticmethod
    def load(file_name):
        """Load a complete model state from a file.
        
        After loading, you can access:
            model.star - The star configuration
            model.prtl - The Particles object with all properties
            model.prtl.matrl - The material properties
        """
        with open(file_name, 'rb') as file:
            return pickle.load(file)