/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Programa;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import weka.classifiers.functions.LinearRegression;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Desconocido
 */
public class Logica {
    
    public ArrayList KMeans(int n) throws FileNotFoundException, IOException, Exception{
        //Indicar la ruta donde está el archivo a leer
        String ruta = "COVID19MX.csv";
        //Cargar el archivo al programa, usando la ruta de este
        Instances dataset = new Instances(new BufferedReader(new FileReader(ruta)));
        //Inicializar un nuevo objeto para aplicar el K-Means
        SimpleKMeans SKM = new SimpleKMeans();
        //Indicar al algoritmo K-Means cuantos clusters se desean
        SKM.setNumClusters(n);
        //Se realiza el agrupamiento
        SKM.buildClusterer(dataset);
        //Almacenar los centroides en un conjuntos de instancias
        Instances c = SKM.getClusterCentroids();
        //Declarar el arraylist que contiene la informacion de los centroides hallados
        ArrayList<Object> i = new ArrayList<>();
        //Almacenar informacion de los centroides en el arraylist 'i'
        for(int x=0;x<n;x++){
            //Indice del centroide
            i.add(x+1);
            //Informacion de dias
            i.add(c.instance(x).value(0));
            //Informacion de casos confirmados
            i.add(c.instance(x).value(1));
            //Informacion de muertes
            i.add(c.instance(x).value(2));
        }
        //Retornar el arraylist 'i'
        return i;
    }
    
    public Instances obtenerDataset(String ruta, int indice) throws FileNotFoundException, IOException{
        //Cargar el archivo al programa, usando la ruta de este
        Instances dataset = new Instances(new BufferedReader(new FileReader(ruta)));
        //Inicializar un FastVector para cargar los atributos
        FastVector fv = new FastVector(2);
        //Agregar el elemento de 'Dias'
        fv.addElement(new Attribute("Casos"));
        if(indice==0){
            //Agregar elemento de casos confirmados, en caso que el indice sea 1
            fv.addElement(new Attribute("Casos Confirmados"));
        }else if(indice==1){
            //Agregar elemento de muertes, en caso que el indice sea 2
            fv.addElement(new Attribute("Muertes"));
        }
        //Declarar dataset
        Instances datasetResultado = new Instances("Dataset",fv,dataset.size());
        //Recorrer los FastVector y el dataset para cargar el 'datasetResultado'
        for(int x=0;x<dataset.size();x++){
            //Declarar una instancia temporal, la cual, luego de ser cargada, será cargada al 'datasetResultado'
            Instance i = new DenseInstance(2);
            //Agregar a la instancia temporal un dato en una posicion [x,0]
            i.setValue((Attribute) fv.elementAt(0), dataset.get(x).value(0));
            //Agregar a la instancia temporal un dato en una posicion [x,1]
            i.setValue((Attribute) fv.elementAt(1), dataset.get(x).value(indice+1));
            //Agregar la instancia temporal a 'datasetResultado'
            datasetResultado.add(i);
        }
        //Seleccionar la clase del dataset que se usará para la clasificacion
        datasetResultado.setClassIndex(1);
        //Retornar el dataset resultado
        return datasetResultado;
    }
    
    public ArrayList regresionLineal(int indice) throws IOException, Exception{
        //Indicar la ruta donde está el archivo a leer
        String ruta = "COVID19MX.csv";
        //Declarar el signo "+" de la ecuacion y=mx+b
        String s="+";
        //Declarar el dataset que será usado para obtener la regresion lineal
        Instances dataset;
        //Obtener el dataset de 2 x N para crear la regresion lineal y guardarlo en la declaracion anterior
        dataset = obtenerDataset(ruta,indice);
        //Inicializar un nuevo objecto para aplicar la regresion lineal
        LinearRegression lr = new LinearRegression();
        //Construir la regresion lineal basada en los datos
        lr.buildClassifier(dataset);
        //Declarar un vector de tipo double, que contenga los coeficientes de la regresion lineal
        double[] c = lr.coefficients();
        //En caso de que el coeficiente sin variable sea negativo, se retira el '+' para dar espacio al signo negativo
        if(c[2]<0){
            s="";
        }
        //Dar forma a la ecuacion, basado en los valores de los coeficientes
        String e = "y="+c[0]+"x"+s+c[2];
        //Declarar un arraylist que contendrá los datos de la ecuacion
        ArrayList<Object>rl = new ArrayList<>();
        //El indice 0 contiene la ecuacion de la forma y=mx+b
        rl.add(e);
        //El indice 1 contiene el coeficiente que acompaña a la x, es decir, m
        rl.add(c[0]);
        //El indice 2 contiene el coeficiente sin variables, es decir, b
        rl.add(c[2]);
        //Retornar el arraylist 'r'
        return rl;
    }
}
