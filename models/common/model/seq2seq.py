import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineSeq2Seq(nn.Module):
    def __init__(self, encode_vocab_size, decode_vocab_size,
                 embed_size, hidden_size):
        super(BaselineSeq2Seq, self).__init__()
        self.encoder_embedding = nn.Embedding(encode_vocab_size, embed_size)
        self.encoder_batchnorm = nn.BatchNorm1d(embed_size)
        self.encoder_gru = nn.GRU(embed_size, hidden_size)
        self.decoder_embedding = nn.Embedding(decode_vocab_size, embed_size)
        self.decoder_batchnorm1 = nn.BatchNorm1d(embed_size)
        self.decoder_gru = nn.GRU(embed_size, hidden_size)
        self.decoder_batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.decoder_linear = nn.Linear(hidden_size, decode_vocab_size)

    def forward_encoder(self, encoder_inputs):
        enc_emb = self.encoder_embedding(encoder_inputs)
        enc_bn = self.encoder_batchnorm(enc_emb.transpose(1, 2)).transpose(1, 2)
        _, state_hidden = self.encoder_gru(enc_bn)
        return state_hidden

    def forward_decoder(self, decoder_inputs, state_hidden):
        dec_emb = self.decoder_embedding(decoder_inputs)
        dec_bn = self.decoder_batchnorm1(dec_emb.transpose(1, 2)).transpose(1, 2)
        output, state_out = self.decoder_gru(dec_bn, state_hidden)
        output = self.decoder_batchnorm2(output.transpose(1, 2)).transpose(1, 2)
        output = self.decoder_linear(output)
        return output, state_out

    def forward(self, encoder_inputs, decoder_inputs):
        state_hidden = self.forward_encoder(encoder_inputs)
        output, _ = self.forward_decoder(decoder_inputs, state_hidden)
        output = F.log_softmax(output, dim=-1)
        return output
