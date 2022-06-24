from copy import deepcopy

def Initial_Broadcast(policy_net,Initial_policy_net,num_workers,model_dict):
    '''
    Initial model update broadcasting operation at server.
    '''
    policy_net.load_state_dict(Initial_policy_net)
    for _ in range(num_workers):
        model_dict['models'].append(deepcopy(policy_net.state_dict()))
        model_dict['targets'].append(deepcopy(policy_net.state_dict()))

    return model_dict


def Register_update(model_dict,widx,model):
    '''
    This operation can be done in main.py.
    '''
    model_dict['models'][widx]=deepcopy(model.state_dict())
    return model_dict['models'][widx]

def FedAvg(Weights,num_worker):
    '''
    Federated Aggregating function.
    '''
    #Peer-to-Peer aggregation
    W1=Weights[0]
    FedAvg_result={}
    for wei in W1:
        FedAvg_result[wei]=W1[wei]*0
        #Clear the variable.
    for widx in range(num_worker):
        for wei in W1:
            temp_weights=Weights[widx]
            FedAvg_result[wei]+= temp_weights[wei]/num_worker

    return FedAvg_result

def Broadcast(Global_model,model_dict,num_workers):
    '''
    Broadcast function between aggregations
    '''
    for widx in range(num_workers):
        for wei in Global_model:
            model_dict['models'][widx][wei]=Global_model[wei]
    return model_dict