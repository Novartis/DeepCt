import torch.nn as nn
import torch


class DeepCt(nn.Module):
    
    def __init__(self, global_feats=0, hidden_feats=None, feat_drops=None, 
                 n_tasks=1, predictor_hidden_feats=128, predictor_dropout=0., num_layers=1):
        super(DeepCt, self).__init__()
        
        # Create FFN layers
        if num_layers == 1:
            ffn = [nn.Dropout(predictor_dropout), nn.Linear(global_feats, n_tasks)]
        else:
            ffn = [nn.Dropout(predictor_dropout), nn.Linear( global_feats, predictor_hidden_feats), nn.BatchNorm1d(predictor_hidden_feats), nn.ReLU()]
            for _ in range(num_layers - 2):
                ffn.extend([nn.Dropout(predictor_dropout), nn.Linear(predictor_hidden_feats, predictor_hidden_feats), nn.BatchNorm1d(predictor_hidden_feats), nn.ReLU()])
            ffn.extend([nn.Linear(predictor_hidden_feats, n_tasks)]);
        
        self.predict = nn.Sequential(*ffn);

    def forward(self, global_feats):
                        
            output = self.predict(global_feats);
            m = nn.Softplus()
            output = m(output)
            
            return output;
