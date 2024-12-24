from roboflow import Roboflow
rf = Roboflow(api_key="TUB3In2YVeE08Dk55sw1")
project = rf.workspace("myfeytech").project("test_project-t9eds")
version = project.version(1)
dataset = version.download("yolov8")
                

