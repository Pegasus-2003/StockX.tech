from tkinter import *
import tkinter.font as tkFont
from tkinter import font
from PIL import ImageTk, Image
import time
import yfinance as yf
from datetime import datetime
from datetime import date
import warnings
import time
from prophet import Prophet
warnings.filterwarnings("ignore")
companyname=""
w = Tk()
w.title("StockX.Tech")
w.resizable(False, False)

window_height = 350
window_width = 600

screen_width = w.winfo_screenwidth()
screen_height = w.winfo_screenheight()

x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))

w.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

def pred(ticker) :
    window = Tk()
    window.title('StockX.Tech')
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (800 / 2))
    y_cordinate = int((screen_height / 2) - (500 / 2))
    window.geometry("{}x{}+{}+{}".format(800, 500, x_cordinate, y_cordinate))
    today=date.today()
    d2 = today.strftime("%B %d, %Y")
    label1 = Label(window , text=ticker)
    label2=Label(window, text=d2)


def stockcurrprice(ticker):
    a = yf.download(tickers=ticker, period='1d', interval='1m')
    a = a.reset_index()
    a = a.filter(['Adj Close'])
    data = a.values
    data = data[len(data) - 1]
    return data


def new_win():

    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    from datetime import datetime
    from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import yfinance as yf
    import math
    from datetime import datetime
    import prophet
    from datetime import datetime
    import yfinance as yf
    from prophet.plot import plot_plotly, plot_components_plotly
    from plotly import graph_objs as go
    import matplotlib.pyplot as plt

    def mainwindow() :
            def sendticker():
                ticker = tickername.get("1.0","end-1c")
                window.destroy()
                secondwindow(ticker)

            window = Tk()

            window_height = 700
            window_width = 982
            window.title('StockX.Tech')

            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()

            x_cordinate = int((screen_width / 2) - (window_width / 2))
            y_cordinate = int((screen_height / 2) - (window_height / 2))
            window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
            window.configure(bg = "#000000")
            window.configure(bg = "#000000")


            canvas = Canvas(
                window,
                bg = "#000000",
                height = 700,
                width = 982,
                bd = 0,
                highlightthickness = 0,
                relief = "ridge"
            )

            canvas.place(x = 0, y = 0)
            canvas.create_rectangle(
                0.0,
                0.0,
                982.0,
                87.0,
                fill="#0B4BEF",
                outline="")


            tickername = Text(

                font=("Comic Sans MS", 35 * -1),
                bd=0,
                bg="#D9D9D9",
                fg="#000716",
                highlightthickness=0
            )
            tickername.place(

                x=224.0,
                y=483.0,
                width=533.0,
                height=50.0
            )

            canvas.create_text(
                330.0,
                7.0,
                anchor="nw",
                text="StockX.Tech",
                fill="#FFFFFF",
                font=("Comic Sans MS", 60 * -1)
            )

            canvas.create_text(
                340.0,
                418.0,
                anchor="nw",
                text="Enter Ticker Name!",
                fill="#0B4BEF",
                font=("Comic Sans MS", 35 * -1)
            )
            button_1=Button(text="Continue",font=("Comic Sans Ms",20 * -1) ,bg="#0B4BEF",width=20,height=2,highlightcolor="#0B4BEF",highlightthickness=2,bd=5,relief="raised",command=sendticker)

            button_1.place(
                x=394.0,
                y=575.0,
                width=212.0,
                height=57.0
            )

            image_image_1 = PhotoImage(
                file=("image_1.png"))
            image_1 = canvas.create_image(
                499.0,
                258.0,
                image=image_image_1
            )
            window.resizable(False, False)
            window.mainloop()
    def secondwindow(ticker) :
        def sendticker():
            start_date = '2012-01-01'
            end_date = datetime.now()
            df = yf.download(ticker, start='2012-01-01', end=datetime.now())
            dataf=df
            df = df.reset_index()
            df1 = df.filter(['Close'])
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df1)
            scaled_data_frame = pd.DataFrame(data=scaled_data)

            stock_close_data = df1.filter(['Close'])
            stock_close_dataset = stock_close_data.values

            trainingDataLength = math.ceil(len(stock_close_dataset) * 0.70)
            scaledData = scaler.fit_transform(stock_close_dataset)

            StockTrainData = scaledData[0:trainingDataLength, :]

            Xtrain = []
            Ytrain = []
            for i in range(60, len(StockTrainData)):
                Xtrain.append(StockTrainData[i - 60:i, 0])
                Ytrain.append(StockTrainData[i, 0])

            Xtrain = np.array(Xtrain)
            Ytrain = np.array(Ytrain)

            Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

            testingData = scaledData[trainingDataLength - 60:, :]

            Xtest = []

            Ytest = stock_close_dataset[trainingDataLength:, :]

            for i in range(60, len(testingData)):
                Xtest.append(testingData[i - 60:i, 0])
            Xtest = np.array(Xtest)
            Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

            from keras.models import load_model

            model = load_model('200epochsensex.h5')


            predictions = model.predict(Xtest)
            predictions = scaler.inverse_transform(predictions)

            training = stock_close_data[:trainingDataLength:]
            validation = pd.DataFrame(df['Close'][trainingDataLength:], columns=['Date', 'Close', 'Predictions'])
            validation['Predictions'] = predictions
            validation['Date'] = df['Date']
            real = validation['Close'].values
            pred = validation['Predictions'].values
            n = len(pred)

            accuracy = 0
            for i in range(n):
                accuracy = (abs(real[i] - pred[i]) / real[i]) * 100

            train = df[:trainingDataLength]
            trainingDates = df['Date'].iloc[:trainingDataLength]
            trainingDates = list(trainingDates.values)
            trainingData = list(training['Close'].values)
            realdata = list(real)
            predictionDates = df['Date'].iloc[trainingDataLength:]
            predictionDates = list(predictionDates.values)
            predictionData = list(pred)
            for i in range(len(trainingData)): trainingData[i] = float(trainingData[i])
            for i in range(len(predictionData)): predictionData[i] = float(predictionData[i])

            plt.figure(figsize=(5.5, 3.7))
            plt.xlabel('Days', fontsize=9)
            plt.ylabel('Close Price', fontsize=9)
            plt.plot(train['Close'])
            plt.plot(validation[['Close', 'Predictions']])
            plt.legend(['Train', 'Original', 'Predictions'], loc='lower right')
            plt.savefig('tested.png')
            

            
            dataf = dataf.reset_index()
            dataf_train = dataf[['Date', 'Close']]
            dataf_train = dataf_train.rename(columns={"Date": "ds", "Close": "y"})
            m = Prophet()
            m.fit(dataf_train)
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)
            fig1 = m.plot(forecast)
            plot_plotly(m, forecast)
            plot_components_plotly(m, forecast)
            plt.gcf().set_size_inches(5.5, 3.7)
            plt.savefig('prophet.png')
            
            window.destroy()
            thirdwindow(ticker)
    
        stock = str(ticker)
        stockinfo = yf.Ticker(stock)
        # print(stockinfo.info['longName'])
        # print(stockinfo.info['sector'])
        # print(stockinfo.info['longBusinessSummary'])
        # print(stockinfo.info['website'])
        # print(stockinfo.info['industry'])
        # print(stockinfo.info['currentPrice'])
        # print(stockinfo.info['financialCurrency'])
        # print(stockinfo.info['market'])

        start_date = "2022-12-01"
        end_date = datetime.now()
        df = yf.download([stock], start=start_date, end=end_date)
        df = df.reset_index()

        df['Date'] = pd.to_datetime(df['Date'])
        new_df = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df1 = df.loc[new_df]
        stock_data = df1.set_index('Date')
        top_plt = plt.subplot2grid((5, 4), (0, 0), rowspan=5, colspan=4, )
        top_plt.plot(stock_data.index,stock_data["Close"], color="blue")
        plt.title(f" Past 1 month stock prices of {stockinfo.info['longName']}")
        plt.gcf().set_size_inches(8, 3)
        plt.savefig('stockmain.png')
        companyname=stockinfo.info['longName']

        from pathlib import Path

        from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

        window = Tk()
        window.title('StockX.Tech')
        window_height = 700
        window_width = 982

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        window.configure(bg="#000000")
        window.configure(bg="#000000")
        window.configure(bg="#000000")

        canvas = Canvas(
            window,
            bg="#000000",
            height=700,
            width=982,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=0)
        canvas.create_rectangle(
            0.0,
            0.0,
            982.0,
            87.0,
            fill="#0B4BEF",
            outline="")

        canvas.create_text(
            350.0,
            100.0,
            anchor="nw",
            text=f"Current Price:{stockinfo.info['currentPrice']} {stockinfo.info['financialCurrency']}",
            fill="#FFFFFF",
            font=("Comic Sans MS", 25 * -1)
        )

        image_image_1 = PhotoImage(master=canvas,
            file="image_12.png")
        image_1 = canvas.create_image(
            51.0,
            43.0,
            image=image_image_1
        )

        canvas.create_text(
            420.0,
            7.0,
            anchor="nw",
            text=stockinfo.info['longName'],
            fill="#FFFFFF",
            font=("Comic Sans MS", 45 * -1)
        )

        image_image_2 = PhotoImage(master=canvas,
            file="stockmain.png")
        image_2 = canvas.create_image(
            500.0,
            450.0,
            image=image_image_2
        )
        button_2 = Button(window,text="Predict", font=("Comic Sans Ms", 20 * -1), bg="#0B4BEF", width=20, height=2,
                          highlightcolor="#0B4BEF", highlightthickness=2, bd=5, relief="raised",command=sendticker)

        button_2.place(
            x=394.0,
            y=620.0,
            width=212.0,
            height=57.0
        )

        canvas.create_text(
            350.0,
            150.0,
            anchor="nw",
            text=f"Sector:{ stockinfo.info['sector']}",
            fill="#FFFFFF",
            font=("Comic Sans MS", 25 * -1)
        )

        canvas.create_text(
            350.0,
            210.0,
            anchor="nw",
            text=f"Industry : {stockinfo.info['industry']}",
            fill="#FFFFFF",
            font=("Comic Sans MS", 25 * -1)
        )

        canvas.create_text(
            350.0,
            265.0,
            anchor="nw",
            text=f"Website : {stockinfo.info['website']}",
            fill="#FFFFFF",
            font=("Comic Sans MS", 25 * -1)
        )
        window.resizable(False, False)
        window.mainloop()

    def thirdwindow(ticker) :
        print("into this")
        window = Tk()

        window_height = 700
        window_width = 1250
        window.title('StockX.Tech')
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        window.configure(bg="#000000")
        

        canvas = Canvas(
            window,
            bg="#000000",
            height=700,
            width=1250,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=0)
        canvas.create_rectangle(
            0.0,
            0.0,
            1250.0,
            87.0,
            fill="#0B4BEF",
            outline="")

        canvas.create_text(
            130.0,
            120.0,
            anchor="nw",
            text="Model Run and Tested",
            fill="#FFFFFF",
            font=("Comic Sans MS", 40 * -1)
        )

        image_image_1 = PhotoImage(master=canvas,
            file=("image_12.png"))
        image_1 = canvas.create_image(
            51.0,
            43.0,
            image=image_image_1
        )

        canvas.create_text(
            500.0,
            7.0,
            anchor="nw",
            text=ticker,
            fill="#FFFFFF",
            font=("Comic Sans MS", 60 * -1)
        )

        image_image_2 = PhotoImage(master=canvas,
            file=("tested.png"))
        image_2 = canvas.create_image(
            300.0,
            420.0,
            image=image_image_2
        )

        image_image_3 = PhotoImage(master=canvas,
            file=("prophet.png"))
        image_3 = canvas.create_image(
            900.0,
            420.0,
            image=image_image_3
        )
        """

        button_1 = Button(window,text="Main Menu", font=("Comic Sans Ms", 20 * -1), bg="#0B4BEF", width=20, height=2,
                          highlightcolor="#0B4BEF", highlightthickness=2, bd=5, relief="raised",command=backwindow)
        button_1.place(
            x=504.0,
            y=725.0,
            width=212.0,
            height=57.0
        )
        """

        canvas.create_text(
            720.0,
            120.0,
            anchor="nw",
            text="Future Predictions",
            fill="#FFFFFF",
            font=("Comic Sans MS", 40 * -1)
        )
        window.resizable(False, False)
        window.mainloop()
    mainwindow()



Frame(w, width=600, height=350, bg='black').place(x=0, y=0)
logoimage = Image.open("logo.png")
resize_image = logoimage.resize((370, 370))
logoimg = ImageTk.PhotoImage(resize_image)
label1 = Label(image=logoimg, borderwidth=0)
label1.pack()
label2 = Label(w, text='Loading...', fg='white', bg='black')
label2.configure(font=("Calibri", 11))
label2.place(x=10, y=315)
image_a = ImageTk.PhotoImage(Image.open('c2.png'))
image_b = ImageTk.PhotoImage(Image.open('c1.png'))
for i in range(2):
    l1 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=270, y=280)
    l2 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=290, y=280)
    l3 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=310, y=280)
    l4 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=330, y=280)
    w.update_idletasks()
    time.sleep(0.1)

    l1 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=270, y=280)
    l2 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=290, y=280)
    l3 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=310, y=280)
    l4 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=330, y=280)
    w.update_idletasks()
    time.sleep(0.1)

    l1 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=270, y=280)
    l2 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=290, y=280)
    l3 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=310, y=280)
    l4 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=330, y=280)
    w.update_idletasks()
    time.sleep(0.1)

    l1 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=270, y=280)
    l2 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=290, y=280)
    l3 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=310, y=280)
    l4 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=330, y=280)
    w.update_idletasks()
    time.sleep(0.1)

w.destroy()
new_win()



w.mainloop()
