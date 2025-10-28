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
  const url1 = "http://192.168.15.182:8000/";
  const url2 = "http://192.168.15.182:8000/";

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
        Alert.alert("Erro", "Nenhum backend disponível!");
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
          Know plant desease with GreenScan AI
        </Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={handleClick}>
        <Text style={styles.buttonText}>Start Scanning</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.button} onPress={testingRequest}>
        <Text style={styles.buttonText}>TESTE</Text>
      </TouchableOpacity>
    </View>
  );
}
