
class ReplayProtector:
    def __init__(self):
        self.last_counter = 0

    def validate(self, counter):
        if counter <= self.last_counter:
            return False
        self.last_counter = counter
        return True
