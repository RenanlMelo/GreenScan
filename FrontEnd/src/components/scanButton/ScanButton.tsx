import { Alert, Text, TouchableOpacity, View } from "react-native";
import { styles } from "./Styles";
import { BrainCircuit } from "lucide-react-native";
import { useNavigation } from "@react-navigation/native";
import type { BottomTabNavigationProp } from "@react-navigation/bottom-tabs";
import axios from "axios";

type TabParamList = {
  Home: undefined;
  Scan: undefined;
  Recents: undefined;
  User: undefined;
};

export default function ScanButton() {
  const navigation = useNavigation<BottomTabNavigationProp<TabParamList>>();
  const url1 = "https://greenscan-uak7.onrender.com";
  const url2 = "http://192.168.15.13:8000/";

  const test = false;

  const testAPI = async (url: string) => {
    const response = await axios.get(url, { timeout: 300 });
    console.log("✅ Resposta do backend:", response.data);
  };

  const testingRequest = async () => {
    try {
      await testAPI(url1);
    } catch (error1) {
      try {
        await testAPI(url2);
      } catch (error2: any) {
        console.error("❌ Erro ao conectar:", error2.message);
        Alert.alert("Erro", "Nenhum backend disponível no momento!");
      }
    }
  };

  function handleClick() {
    navigation.navigate("Scan");
  }

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <View style={styles.icon}>
          <BrainCircuit stroke="#B2D5B8" size={28} />
        </View>
        <Text style={styles.scanText}>
          Identifique doenças nas plantas com a GreenScan AI
        </Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={handleClick}>
        <Text style={styles.buttonText}>Iniciar análise</Text>
      </TouchableOpacity>

      {test && (
        <TouchableOpacity style={styles.button} onPress={testingRequest}>
          <Text style={styles.buttonText}>TESTAR CONEXÃO</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}
