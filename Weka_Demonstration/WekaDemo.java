import java.awt.*;
import java.lang.reflect.Field;
import java.nio.file.*;
import javax.swing.*;
import javax.imageio.ImageIO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.AbstractClassifier;
import weka.gui.GenericObjectEditor;
import weka.gui.explorer.Explorer;
import weka.gui.explorer.ClassifierPanel;

/** Opens the actual Weka Explorer. Optional --capture runs it and records real GUI screenshots. */
public class WekaDemo {
  static Object field(Object o,String name)throws Exception {
    Field f=ClassifierPanel.class.getDeclaredField(name);f.setAccessible(true);return f.get(o);
  }
  static void capture(JFrame f,Path p)throws Exception {
    f.toFront();Toolkit.getDefaultToolkit().sync();
    ImageIO.write(new Robot().createScreenCapture(f.getBounds()),"png",p.toFile());
  }
  public static void main(String[] args)throws Exception {
    final Path root=Paths.get(args[0]).toAbsolutePath();
    final boolean record=args.length>1&&args[1].equals("--capture");
    final Instances data=DataSource.read(root.resolve("data/Ionosphere_development.arff").toString());
    data.setClassIndex(data.numAttributes()-1);
    SwingUtilities.invokeLater(()->{try {
      Explorer explorer=new Explorer();
      JFrame frame=new JFrame("Weka Explorer - CDS535 Group - Ionosphere demonstration");
      frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);frame.setContentPane(explorer);
      Dimension screen=Toolkit.getDefaultToolkit().getScreenSize();
      frame.setSize(Math.min(1280,screen.width-40),Math.min(900,screen.height-70));frame.setLocation(15,15);frame.setVisible(true);
      explorer.getPreprocessPanel().setInstances(data);
      ClassifierPanel panel=null;
      for(Explorer.ExplorerPanel p:explorer.getPanels())if(p instanceof ClassifierPanel)panel=(ClassifierPanel)p;
      final ClassifierPanel cp=panel;
      String filters="weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.ReplaceMissingValues\" -F \"weka.filters.unsupervised.attribute.NominalToBinary\" -F \"weka.filters.unsupervised.attribute.RemoveUseless -M 100.0\" -F \"weka.filters.unsupervised.attribute.Standardize\"";
      ((GenericObjectEditor)field(cp,"m_ClassifierEditor")).setValue(AbstractClassifier.forName("weka.classifiers.meta.FilteredClassifier",new String[]{"-F",filters,"-W","weka.classifiers.trees.J48","--","-C","0.25","-M","2"}));
      ((JRadioButton)field(cp,"m_CVBut")).doClick();
      ((JTextField)field(cp,"m_CVText")).setText("10");
      ((JTextField)field(cp,"m_RandomSeedText")).setText("535");
      if(!record){explorer.getTabbedPane().setSelectedIndex(1);return;}
      Timer initial=new Timer(2500,null);initial.addActionListener(e->{try{
        initial.stop();capture(frame,root.resolve("screenshots/01_preprocess.png"));
        explorer.getTabbedPane().setSelectedIndex(1);
        Timer run=new Timer(1500,null);run.addActionListener(e2->{try{
          run.stop();capture(frame,root.resolve("screenshots/02_classifier_setup.png"));
          ((JButton)field(cp,"m_StartBut")).doClick();
          Timer poll=new Timer(1000,null);poll.addActionListener(e3->{try{
            JTextArea output=(JTextArea)field(cp,"m_OutText");
            if(field(cp,"m_RunThread")==null&&output.getText().contains("Confusion Matrix")){
              poll.stop();Files.writeString(root.resolve("results/Ionosphere_J48_Explorer_output.txt"),output.getText());
              output.setCaretPosition(output.getText().length());
              Timer shot=new Timer(1200,null);shot.addActionListener(e4->{try{shot.stop();capture(frame,root.resolve("screenshots/03_evaluation_result.png"));System.out.println("GUI_CAPTURE_COMPLETE");}catch(Exception ex){ex.printStackTrace();}});shot.start();
            }
          }catch(Exception ex){ex.printStackTrace();poll.stop();}});poll.start();
        }catch(Exception ex){ex.printStackTrace();}});run.start();
      }catch(Exception ex){ex.printStackTrace();}});initial.start();
    }catch(Exception ex){ex.printStackTrace();}});
  }
}
