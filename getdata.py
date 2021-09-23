import random 
def get_data_custom_multidimIO(X_data, y_data, batch_size, in_seq_len, out_seq_len):
  
    seq_len = in_seq_len + out_seq_len
    if X_data.ndim==1:
      whole_time_len = len(X_data)
    else:
      whole_time_len = len(X_data[0,:])
      
    n_seq = int( whole_time_len/ seq_len )
    if X_data.ndim == 1:
        n_x_features = 1
    else:
        n_x_features = x_data.shape[0]

    if y_data.ndim == 1:
        n_y_features = 1
    else:
        n_y_features = y_data.shape[0]
    X, y = [], []
    
    #sequnenceのスタートする時間indexを out_seq_lenずつずらす
    all_strart_points = [i for i in range(0 ,whole_time_len - seq_len,  out_seq_len)]

    #np.random.choice(a, size=None, replace=True, p=None)
    #batchsizeが小さい時はsequenceがかぶらないようにreplace=Falceでchoice,大きい時はsequenceの被りが発生するのでreplace=True

    #random.seed(0) #乱数のシード
    #重複あり
    if len( all_strart_points ) < batch_size :
      start_point_list = random.choices(all_strart_points, k=batch_size)
    #重複なし
    else :
      start_point_list = random.sample(all_strart_points, batch_size)

    print("start_point_list =", start_point_list)
    if X_data.ndim==1 and y_data.ndim==1:  
        for start_point in start_point_list :
            X.append(X_data[start_point : start_point + in_seq_len])
            y.append(y_data[start_point+ in_seq_len : start_point+ seq_len])
        #return torch.tensor(X).unsqueeze(-1).float(),  torch.tensor(y).unsqueeze(0).float()

    elif X_data.ndim !=1 and y_data.ndim ==1:  
        for start_point in start_point_list :
            X.append(X_data[ : , start_point : start_point + in_seq_len])
            y.append(y_data[start_point+ in_seq_len : start_point + seq_len])
        #X = torch.reshape(torch.tensor(X).float(), (batch_size , enc_seq_len, n_x_features))
        #y = torch.reshape(torch.tensor(y).float(), (batch_size , output_sequence_length))
    
    elif X_data.ndim ==1 and y_data.ndim !=1:  
        for start_point in start_point_list :
            X.append(X_data[ start_point : start_point + in_seq_len])
            y.append(y_data[: , start_point+ in_seq_len : start_point + seq_len])
        #return torch.tensor(X).unsqueeze(-1).float(),  torch.tensor(y).float()

    else: 
        for start_point in start_point_list:
            #########ここのスライシングで(batch_size, enc_seq_len, input_size)にする#####
            X.append(X_data[ : , start_point : start_point + in_seq_len])
            y.append(y_data[ : , start_point + in_seq_len : start_point + seq_len])

    X = torch.reshape(torch.tensor(X).float(), (batch_size , enc_seq_len, n_x_features))
    y = torch.reshape(torch.tensor(y).float(), (batch_size , output_sequence_length,  n_y_features))
    return X, y 
