from pprint import pprint

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()

service = discovery.build('compute', 'v1', credentials=credentials)

# Project ID for this request.
project = 'burnished-road-266221'  # TODO: Update placeholder value.

instance_move_request_body = {
    # TODO: Add desired entries to the request body.

}

request = service.projects().moveInstance(project=project, body=instance_move_request_body)
response = request.execute()

# TODO: Change code below to process the `response` dict:
pprint(response)
