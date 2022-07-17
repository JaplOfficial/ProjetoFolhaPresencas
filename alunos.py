import cv2
import os
import utils
import imutils
import numpy as np
import folhaPresenca
import pytesseract
import assinaturas_validar
import statistics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


IMGX = utils.IMGX
IMGY = utils.IMGY


def encontraNumeroAluno(imgAluno):
    #encontra especificamente onde o número de aluno se encontra.
    src = cv2.cvtColor(imgAluno, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(src, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dst = cv2.GaussianBlur(thresh,(5,5),cv2.BORDER_DEFAULT)
    notGray = cv2.bitwise_not(dst)

    #Aplica um close ao nr de aluno
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    max = cv2.contourArea(contours[0])
    b = contours[0]
    for c in contours:
        area = cv2.contourArea(c)
        if area > max:
            max = area
            b = c

    (x, y, w, h) = cv2.boundingRect(b)

    if (x != 0 and y != 0):
        roi = imgAluno[y: y + h, x: x + w]
    else:
        roi = imgAluno[y: y + h + 2, x: x + w + 3]
    return roi


def confirmaAssinatura(assinatura, turma, nAluno):
    #Confirma se realmente existe uma assinatura, através do seu tamanho
    assinaturaGrey = cv2.cvtColor(assinatura, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(assinaturaGrey)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (10, 5))
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    maxW = 0
    coord = None
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > maxW:
            coord = (x, y, w, h)
            maxW = w

    return coord


def verificaAssinatura(assinatura, X_train, y_train, mlp, turma, nAluno):
    nAluno = nAluno.rstrip('\n')
    try:
        if(os.path.isdir("./turmas/" + turma + "/" + nAluno) == False):
            print('maybe')
            os.mkdir("./turmas/" + turma + "/" + nAluno)
            os.mkdir("./turmas/" + turma + "_augmented/" + nAluno)
        list = os.listdir("./turmas/" + turma + "/" + nAluno)
        number_files = len(list)
        #print('number of files: ' +str(number_files))
        a = mlp.predict(vector_img)
        if (a == 1): #so acrescenta se nao for uma falta
            cv2.imwrite("./turmas/" + str(turma) + "/" + str(nAluno) + "/" + str(number_files) + ".jpg",assinatura)
    except Exception as e:
    # ... PRINT THE ERROR MESSAGE ... #
        print(e)

    assinado = 0
    #recorre as transformacoes da imagem
    img_trf = utils.img_transformation(assinatura)

    # percorre a imagem da assinatura e soma as linhas e as colunas
    sum_of_rows = np.sum(img_trf, axis = 1)
    sum_of_columns = np.sum(img_trf, axis = 0)

    #concatena as somas e transpoe para vector
    vector_img = np.concatenate((sum_of_rows, sum_of_columns))

    vector_img = np.array(vector_img)[np.newaxis]
    #print("DEPOIS TRANPOSE")
    #print(vector_img.shape)

    # standardiza os valores das somas
    vector_img = StandardScaler().fit_transform(vector_img)

    #carrega o modelo
    #mlp.fit(X_train, y_train)
    assinado = mlp.predict(vector_img)

    #devolve a label atribuida
    print(assinado)

    return assinado



def extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas):
    #extrai linhas horizontais das tabelas para ser possível individualizar cada aluno
    (x, y, w, h) = roiAlunosLinhas
    roi = linhasHorizontais[y:y + h, x:x + w]
    linhasAlunos = []
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 30, minLineLength=400, maxLineGap=250)

    if len(lines) > 0:
        for line in lines:
            (x1, y1, x2, y2) = line[0]
            linhasAlunos.append((x1, y1, x2, y2))

        linhasAlunos = sorted(linhasAlunos, key=lambda y: y[1])

        linhasAlunosFinal = []
        (_, y, _, _) = linhasAlunos[0]
        for linha in linhasAlunos:
            (x1, y1, x2, y2) = linha
            if y1 - y > 10:
                linhasAlunosFinal.append(linha)
                y = y1
    else:
        quit("Folha ilegível, falha na divisão dos alunos")
    return linhasAlunosFinal


def processaAlunos(img, X_train, y_train, mlp, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas, codigoAula, turma):
    todosAlunos = []
    alunosPresentes = []
    folhaInvalida = 0
    contador = 1



    linhasHorizontais, linhas = folhaPresenca.filtroDeLinhas(img)
    roiAlunos, roiAlunosLinhas = folhaPresenca.encontraTabelasAlunos(img, linhas)

    # tesseract configuration
    custom_config = r'--oem 3 --psm 6 outputbase digits'

    #percorre as tabelas dos alunos encontradas
    for r in range(len(roiAlunos)):

        linhasAlunos = extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas[r])
        larg = roiAlunos[r].shape[1]

        for i in range(len(linhasAlunos) - 1):
            (x1, Yi, x2, y2) = linhasAlunos[i]
            (x1, y1, x2, Yf) = linhasAlunos[i + 1]
            altura = Yf - Yi
            if altura < int(IMGY * 0.005):
                continue
            #extrai um aluno consoante as linhas
            Aluno = roiAlunos[r][Yi:Yf, 5:larg - 10]
            #extrai número de aluno consoante coordenadas fixas.
            nrAluno = Aluno[int(round(altura * 0.1)):int(round(altura * 0.93)),
                int(round(larg * 0.1)):int(round(larg * 0.29))]

            #aproximação exata ao local do número de aluno
            nrAluno = encontraNumeroAluno(nrAluno)
            nrAlunoGray = cv2.cvtColor(nrAluno, cv2.COLOR_BGR2GRAY)

            pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR/tesseract.exe'
            # utiliza o pytesseract para ler o id do aluno
            #print('hmmmmmmmm')
            n_out = pytesseract.image_to_string(nrAlunoGray, config=custom_config)



            #remove espaco na string obtida por ocr
            #remove pontos finais na string obtida por ocr
            n_clean = n_out.replace(' ','').replace('.','')


            # pega no output do tesseract e retorna uma list com digitos apenas
            # a list e transformada em string e tem de ter tamanho 10
            n = ",".join([str(s) for s in n_clean.split() if s.isdigit()])

            # se tiver tam 10 entao faz cast para inteiro
            if len(n) == 10:
                try:
                    numero_Aluno = int(n)
                except ValueError:
                    print("*****************************************************************")
                    print("Não foi possível converter o número do aluno corretamente")
                    print("*****************************************************************")
                    out_ilegivel += 1
                    numero_Aluno = 0

                assinatura = Aluno[int(round(altura * 0.25)):int(round(altura * 0.94)),
                                     int(round(larg * 0.74)):int(round(larg * 0.97))]
                #usa o classificador para ver se assinatura esta presente
                assinado = verificaAssinatura(assinatura, X_train, y_train, mlp, turma, n_clean)
                todosAlunos.append(numero_Aluno)
                if assinado:
                    # comentado porque faz sentido apenas se ler uma folha de cada vez
                    # ff.folhaFinal(imgFinal, roiAlunosLinhas[r], Yi, imgCerto)
                    alunosPresentes.append(numero_Aluno)
                    print(str(contador) + " - " + str(numero_Aluno) + " = PRESENTE")
                    out_presente += 1
                else:
                    print(str(contador) + " - " + str(numero_Aluno))
                    out_ausente += 1
                contador += 1

            else:
                try:
                    numero_Aluno = int(n)
                    print(str(contador) + " - Não foi possível ler o número do aluno corretamente -> " + str(numero_Aluno))
                except ValueError:
                    print(str(contador) + " - Erro na leitura do número -> NULL ")
                out_erro_num += 1
                contador += 1
        if folhaInvalida > len(todosAlunos):
            out_problemas += 1
            quit("folha com problemas")

    count_alunos += len(todosAlunos)
    return todosAlunos, alunosPresentes, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas
