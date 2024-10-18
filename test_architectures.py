
import models as mo


mo.test_model(mo.cnn,"original")

mo.test_model(mo.cnn,"audio")

mo.test_model(mo.cnn,"img")


mo.test_model(mo.cnn_lstm,"original")

mo.test_model(mo.cnn_lstm,"audio")

mo.test_model(mo.cnn_lstm,"img")


mo.test_model(mo.cnn_lstm_attention,"original") 

mo.test_model(mo.cnn_lstm_attention,"audio") 

mo.test_model(mo.cnn_lstm_attention,"img")
