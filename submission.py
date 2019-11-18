import requests
import json


def submission(payload, is_test=False, host='192.168.1.200', port='3000'):

    if is_test:
        url = "http://" + host + ":" + port + "/submit"
    else:
        url = "http://" + host + ":" + port + "/submit"

    header = {
        "Content-Type": "application/json",
        "cache-control": "no-cache",
        "Authorization": "Bearer " + "NA3TTAB-ET7MF2J-P21KZY2-A3XK35N"
    }

    response_decoded_json = requests.post(
        url, data=json.dumps(payload), headers=header)
    try:
        response_json = response_decoded_json.json()
        print("Complete to request", payload)
        print()
        # f = open('./scene_no-' + str(payload['scene_no']+".txt"),'w+')
        # f.write(str(payload)+"\n")
        # f.write(str(response_json))
        # f.close()
        return True
    except:
        print("Cannot submit the answer")
        print()
        return False

# payload = {
#     "scene_no": 2,
#     "ppe": {
#         "helmet": True,
#         "glasses": True,
#         "coverall": False,
#         "boots": False,
#         "gloves": False
#     }
# }
# submission(payload,is_test=False)
