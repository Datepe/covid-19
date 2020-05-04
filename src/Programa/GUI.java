/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Programa;

import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JOptionPane;
import javax.swing.table.DefaultTableModel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.core.Instances;

/**
 *
 * @author Desconocido
 */
public class GUI extends javax.swing.JFrame {
    //Declarar el salto de linea que será usado en los textos a lo largo de programa
    String sl = System.getProperty("line.separator");
    //Declarar un objecto que permita el uso de la logica del programa
    Logica l = new Logica();
    //Declarar el modelo que seguirá la tabla
    DefaultTableModel tablaModelo;
    /**
     * Creates new form Centoides
     */
    public GUI() {
        initComponents();
        //Hacer invisible la tabla para menor carga grafica en la GUI
        tablaCentroides.setVisible(false);
        //Declarar el vector que contiene la cabecera de la tabla
        String tablaCabecera[]={"Indice de centroide","Dias","Casos","Muertes"};
        //Declarar la matriz que usa la tabla para mostrar los datos
        String tablaDatos[][]={};
        //Inicializar el modelo de la tabla con los datos que contendrá esta y la cabecera
        tablaModelo = new DefaultTableModel(tablaDatos, tablaCabecera);
        //Establecer que el modelo que usará la tabla es el programado
        tablaCentroides.setModel(tablaModelo);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        pestaniasOpciones = new javax.swing.JTabbedPane();
        kMeans = new javax.swing.JPanel();
        panelTabla = new javax.swing.JScrollPane();
        tablaCentroides = new javax.swing.JTable();
        s1 = new javax.swing.JSeparator();
        t1 = new javax.swing.JLabel();
        nCentroides = new javax.swing.JTextField();
        obtenerCentroides = new javax.swing.JButton();
        regresionLineal = new javax.swing.JPanel();
        t2 = new javax.swing.JLabel();
        opciones = new javax.swing.JComboBox<>();
        obtenerInformacion = new javax.swing.JButton();
        s2 = new javax.swing.JSeparator();
        t3 = new javax.swing.JLabel();
        ecuacionRegresion = new javax.swing.JTextField();
        s3 = new javax.swing.JSeparator();
        t4 = new javax.swing.JLabel();
        t5 = new javax.swing.JLabel();
        diasPronostico = new javax.swing.JTextField();
        obtenerPronostico = new javax.swing.JButton();
        t6 = new javax.swing.JLabel();
        casosPronostico = new javax.swing.JTextField();
        opcion = new javax.swing.JLabel();
        sVertical = new javax.swing.JSeparator();
        panelGrafico = new javax.swing.JPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setResizable(false);

        tablaCentroides.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {

            },
            new String [] {
                "Indice. de Centroide", "Dias", "Casos", "Muertes"
            }
        ) {
            boolean[] canEdit = new boolean [] {
                false, false, true, false
            };

            public boolean isCellEditable(int rowIndex, int columnIndex) {
                return canEdit [columnIndex];
            }
        });
        tablaCentroides.setEnabled(false);
        panelTabla.setViewportView(tablaCentroides);
        if (tablaCentroides.getColumnModel().getColumnCount() > 0) {
            tablaCentroides.getColumnModel().getColumn(0).setResizable(false);
            tablaCentroides.getColumnModel().getColumn(1).setResizable(false);
            tablaCentroides.getColumnModel().getColumn(2).setResizable(false);
            tablaCentroides.getColumnModel().getColumn(3).setResizable(false);
        }

        t1.setText("Ingrese el numero de centroides a encontrar:");

        obtenerCentroides.setText("Obtener centroides");
        obtenerCentroides.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                obtenerCentroidesActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout kMeansLayout = new javax.swing.GroupLayout(kMeans);
        kMeans.setLayout(kMeansLayout);
        kMeansLayout.setHorizontalGroup(
            kMeansLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(kMeansLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(kMeansLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(panelTabla, javax.swing.GroupLayout.DEFAULT_SIZE, 664, Short.MAX_VALUE)
                    .addComponent(s1)
                    .addGroup(kMeansLayout.createSequentialGroup()
                        .addComponent(t1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(nCentroides, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(obtenerCentroides)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        kMeansLayout.setVerticalGroup(
            kMeansLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, kMeansLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(kMeansLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(t1)
                    .addComponent(nCentroides, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(obtenerCentroides))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(s1, javax.swing.GroupLayout.PREFERRED_SIZE, 10, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(panelTabla, javax.swing.GroupLayout.DEFAULT_SIZE, 200, Short.MAX_VALUE)
                .addContainerGap())
        );

        pestaniasOpciones.addTab("Obtener Centroides", kMeans);

        t2.setText("Seleccione la informacion que desea encontrar:");

        opciones.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Dias-Casos", "Dias-Muertes" }));
        opciones.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                opcionesActionPerformed(evt);
            }
        });

        obtenerInformacion.setText("Obtener informacion");
        obtenerInformacion.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                obtenerInformacionActionPerformed(evt);
            }
        });

        t3.setText("Regresion lineal:");

        ecuacionRegresion.setEditable(false);

        t4.setText("Herramienta de pronostico:");

        t5.setText("Dia del pronostico:");

        obtenerPronostico.setText("Obtener pronostico");
        obtenerPronostico.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                obtenerPronosticoActionPerformed(evt);
            }
        });

        t6.setText("Para el dia seleccionado se estiman");

        casosPronostico.setEditable(false);

        opcion.setText("casos   ");

        sVertical.setOrientation(javax.swing.SwingConstants.VERTICAL);

        panelGrafico.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        javax.swing.GroupLayout panelGraficoLayout = new javax.swing.GroupLayout(panelGrafico);
        panelGrafico.setLayout(panelGraficoLayout);
        panelGraficoLayout.setHorizontalGroup(
            panelGraficoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 323, Short.MAX_VALUE)
        );
        panelGraficoLayout.setVerticalGroup(
            panelGraficoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout regresionLinealLayout = new javax.swing.GroupLayout(regresionLineal);
        regresionLineal.setLayout(regresionLinealLayout);
        regresionLinealLayout.setHorizontalGroup(
            regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(regresionLinealLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(s2, javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(s3, javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(t2)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, regresionLinealLayout.createSequentialGroup()
                        .addComponent(opciones, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGap(18, 18, 18)
                        .addComponent(obtenerInformacion))
                    .addComponent(t3)
                    .addComponent(t4)
                    .addGroup(regresionLinealLayout.createSequentialGroup()
                        .addComponent(t5)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(diasPronostico, javax.swing.GroupLayout.PREFERRED_SIZE, 70, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(obtenerPronostico, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(ecuacionRegresion)
                    .addGroup(regresionLinealLayout.createSequentialGroup()
                        .addComponent(t6)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(casosPronostico, javax.swing.GroupLayout.PREFERRED_SIZE, 89, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(opcion)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(sVertical, javax.swing.GroupLayout.PREFERRED_SIZE, 10, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(panelGrafico, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        regresionLinealLayout.setVerticalGroup(
            regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(regresionLinealLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(sVertical)
                    .addGroup(regresionLinealLayout.createSequentialGroup()
                        .addComponent(t2)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addGroup(regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(opciones, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(obtenerInformacion))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(s2, javax.swing.GroupLayout.PREFERRED_SIZE, 2, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(13, 13, 13)
                        .addComponent(t3)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(ecuacionRegresion, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(s3, javax.swing.GroupLayout.PREFERRED_SIZE, 2, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(t4)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addGroup(regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(t5)
                            .addComponent(diasPronostico, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(obtenerPronostico))
                        .addGap(18, 18, 18)
                        .addGroup(regresionLinealLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(t6)
                            .addComponent(casosPronostico, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(opcion))
                        .addGap(0, 21, Short.MAX_VALUE))
                    .addComponent(panelGrafico, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );

        pestaniasOpciones.addTab("Regresion Lineal", regresionLineal);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(pestaniasOpciones, javax.swing.GroupLayout.PREFERRED_SIZE, 689, javax.swing.GroupLayout.PREFERRED_SIZE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(pestaniasOpciones)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    
    private void obtenerGrafica() throws IOException{
        //Declarar un string donde se almacenará la opcion seleccionada
        String o ="";
        //Validar la opcion seleccionada
        if(opciones.getSelectedIndex()==0){
            o = "Casos confirmados";
        }else if(opciones.getSelectedIndex()==1){
            o = "Muertes";
        }
        //Obtener el dataset que alimenta con datos a la grafica
        Instances i = l.obtenerDataset("COVID19MX.csv", opciones.getSelectedIndex());
        //Declarar una nueva serie, que contendrá los datos
        XYSeries xys = new XYSeries("Escala Dias-"+o);
        //Cargar la serie con los datos del dataset
        for(int x=0;x<i.size();x++){
            xys.add(i.get(x).value(0), i.get(x).value(1));
        }
        //Declarar una nueva coleccion de series
        XYSeriesCollection c = new XYSeriesCollection();
        //Agregar la serie creada a la coleccion
        c.addSeries(xys);
        //Declarar un nuevo grafico con sus respectivos atributos
        JFreeChart g = ChartFactory.createXYLineChart("Dias-"+o, "Dias", o, c,PlotOrientation.VERTICAL,true, false, false);
        //Declarar el panel del grafico, cargado con los atributos del grafico mismo
        ChartPanel gp = new ChartPanel(g);
        //Configurar el panel que se usará para mostrar el grafico
        panelGrafico.setLayout(new java.awt.BorderLayout());
        //Agregar panel de grafico al panel original
        panelGrafico.add(gp);
        //Validar ejecucion del panel
        panelGrafico.validate();
    }
    
    private void obtenerCentroidesActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_obtenerCentroidesActionPerformed
        try{
            //Verificar que el numero de centroides a obtener no sea negativo o nulo
            if((Integer.parseInt(nCentroides.getText()))<=0){
                //Mostrar mensaje de error al usuario en dicho caso
                JOptionPane.showMessageDialog(null, "No se puede obtener un numero nulo/negativo"+sl+"de centroides. Solo valores positivos");
            }else{
                //Borrar datos que haya tenido la tabla anteriormente
                tablaModelo.setRowCount(0);
                //Declarar el arraylist que contendrá la informacion de los centroides
                ArrayList<String> c = new ArrayList<>();
                try {
                    //Contener la informacion de los centroides
                    c = l.KMeans(Integer.parseInt(nCentroides.getText()));
                    //Recorrer la informacion de los centroides
                    for(int x=0;x<c.size();x=x+4){
                        //Declarar un vector con la informacion que ira a la tabla
                        String datos[]={String.valueOf(c.get(x)),String.valueOf(c.get(x+1)),String.valueOf(c.get(x+2)),String.valueOf(c.get(x+3))};
                        //Agregar vector al modelo y, por ende, a la tabla
                        tablaModelo.addRow(datos);
                    }
                    //Hacer la tabla visible, con sus respectivos datos
                    tablaCentroides.setVisible(true);
                } catch (Exception ex) {
                    Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }catch(Exception e){
            JOptionPane.showMessageDialog(null,"Solo se permiten numeros");
        }
    }//GEN-LAST:event_obtenerCentroidesActionPerformed

    private void obtenerInformacionActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_obtenerInformacionActionPerformed
        //Declarar el arraylist que contiene los datos de la regresion lineal
        ArrayList<Object> rl = new ArrayList<>();
        try {
            //Contener los datos de la regresion lineal en el arraylist creado
            rl = l.regresionLineal(opciones.getSelectedIndex());
        } catch (Exception ex) {
            Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
        }
        //Establecer la ecuacion obtenida en el espacio designado
        ecuacionRegresion.setText((String) rl.get(0));
        //Activar la generacion de la grafica
        try {
            obtenerGrafica();
        } catch (IOException ex) {
            Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_obtenerInformacionActionPerformed

    private void obtenerPronosticoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_obtenerPronosticoActionPerformed
        try{
            //Declarar el arraylist que contiene los datos de la regresion lineal
            ArrayList<Object> rl = new ArrayList<>();
            //Obtener en formato numerico el dato
            double x = Integer.parseInt(diasPronostico.getText());
            try {
                //Realizar la regresion lineal solicitada, para obtener los datos necesarios
                rl=l.regresionLineal(opciones.getSelectedIndex());

            } catch (Exception ex) {
                Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
            }
            //Almancenar el primer coeficiente
            double m  = (double) rl.get(1);
            //Almancenar el segundo coeficiente
            double b = (double) rl.get(2);
            //Realizar la operacion
            double y = m*x+b;
            //Establecer la validacion en caso de que el resultado sea negativo
            if(y<0){
                y=0;
            }
            //Establecer el resultado en el espacio designado
            casosPronostico.setText(String.valueOf(y));
        }catch(Exception e){
            JOptionPane.showMessageDialog(null,"Solo se permiten numeros");
        }
        
    }//GEN-LAST:event_obtenerPronosticoActionPerformed

    private void opcionesActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_opcionesActionPerformed
        //Cambiar el texto auxiliar de la prediccion dependiendo de  la opcion seleccionada
        if(opciones.getSelectedIndex()==0){
            opcion.setText("casos  ");
        }else if(opciones.getSelectedIndex()==1){
            opcion.setText("muertes");
        }
    }//GEN-LAST:event_opcionesActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Windows".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(GUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new GUI().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField casosPronostico;
    private javax.swing.JTextField diasPronostico;
    private javax.swing.JTextField ecuacionRegresion;
    private javax.swing.JPanel kMeans;
    private javax.swing.JTextField nCentroides;
    private javax.swing.JButton obtenerCentroides;
    private javax.swing.JButton obtenerInformacion;
    private javax.swing.JButton obtenerPronostico;
    private javax.swing.JLabel opcion;
    private javax.swing.JComboBox<String> opciones;
    private javax.swing.JPanel panelGrafico;
    private javax.swing.JScrollPane panelTabla;
    private javax.swing.JTabbedPane pestaniasOpciones;
    private javax.swing.JPanel regresionLineal;
    private javax.swing.JSeparator s1;
    private javax.swing.JSeparator s2;
    private javax.swing.JSeparator s3;
    private javax.swing.JSeparator sVertical;
    private javax.swing.JLabel t1;
    private javax.swing.JLabel t2;
    private javax.swing.JLabel t3;
    private javax.swing.JLabel t4;
    private javax.swing.JLabel t5;
    private javax.swing.JLabel t6;
    private javax.swing.JTable tablaCentroides;
    // End of variables declaration//GEN-END:variables
}