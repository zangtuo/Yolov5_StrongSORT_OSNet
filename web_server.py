from flask import Flask, request

app = Flask(__name__)

@app.route('/api/submit/cam/pedestrian', methods=['POST'])
def submitCamPed():
    request_body = request.get_json()
    print(request_body)
    return {
        'result':'ok',
        'message':{
            'tid': 0,
            'state': 'processing'
        }
    }

@app.route('/api/submit/drone/pedestrian', methods=['POST'])
def submitDronePed():
    request_body = request.get_json()
    print(request_body)
    return {
        'result':'ok',
        'message':{
            'tid': 0,
            'state': 'processing'
        }
    }

@app.route('/api/submit/drone/vehicle', methods=['POST'])
def submitDroneVehicle():
    request_body = request.get_json()
    print(request_body)
    return {
        'result':'ok',
        'message':{
            'tid': 0,
            'state': 'processing'
        }
    }

app.run(host="0.0.0.0", port=9005)
