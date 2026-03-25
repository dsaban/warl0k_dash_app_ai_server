
events=[]
def log_event(t,d): events.append({"type":t,"data":d})
def get_events(): return events
