import React, { useState } from "react";
import {
  View,
  Image,
  TouchableOpacity,
  Text,
  ActivityIndicator,
} from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { styles } from "./Styles";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: { reportId: number };
};

type Props = NativeStackScreenProps<RootStackParamList, "Preview">;

const API_URL = "http://192.168.15.13:8000";

export function PreviewScreen({ route, navigation }: Props) {
  const { photoUri } = route.params;
  const [loading, setLoading] = useState(false);

  async function sendPhoto() {
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("image", {
        uri: photoUri,
        name: "photo.jpg",
        type: "image/jpeg",
      } as any);

      const response = await fetch(`${API_URL}/ai/analyze`, {
        method: "POST",
        body: formData,
        headers: { "Content-Type": "multipart/form-data" },
      });
      const reportData = await response.json();
      navigation.replace("Report", { reportId: reportData.data.id });
    } catch (error) {
      console.error("Erro ao enviar foto:", error);
      setLoading(false);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: "black" }}>
      {/* Imagem ocupando toda a tela */}
      <Image
        source={{ uri: photoUri }}
        style={{ flex: 1, resizeMode: "cover" }}
      />

      {/* Botões */}
      <View
        style={{
          position: "absolute",
          bottom: 32,
          width: "100%",
          flexDirection: "row",
          justifyContent: "space-around",
          alignItems: "center",
        }}
      >
        {/* Tirar outra */}
        <TouchableOpacity
          style={{
            width: 75,
            height: 75,
            borderRadius: 50,
            backgroundColor: "#303030aa",
            justifyContent: "center",
            alignItems: "center",
          }}
          onPress={() => navigation.goBack()}
        >
          <MaterialCommunityIcons
            name="camera-retake"
            size={32}
            color="white"
          />
        </TouchableOpacity>

        {/* Confirmar */}
        <TouchableOpacity
          style={{
            width: 70,
            height: 70,
            borderRadius: 35,
            backgroundColor: "#00AA00",
            justifyContent: "center",
            alignItems: "center",
          }}
          onPress={sendPhoto}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <MaterialCommunityIcons name="check" size={36} color="white" />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}
