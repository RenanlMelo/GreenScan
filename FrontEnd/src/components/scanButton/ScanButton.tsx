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

  const testAPI = async () => {
    try {
      const response = await axios.get("http://192.168.15.13:8000/");
      console.log("✅ Resposta do backend:", response.data);
    } catch (error: any) {
      console.error("❌ Erro:", error.message);
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
      <TouchableOpacity style={styles.button} onPress={testAPI}>
        <Text style={styles.buttonText}>TESTE</Text>
      </TouchableOpacity>
    </View>
  );
}
