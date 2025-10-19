import React from "react";
import { View, Image, TouchableOpacity, Alert } from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import axios from "axios";
import { styles } from "./Styles";
import { NativeStackScreenProps } from "@react-navigation/native-stack";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: {
    photoUri: string;
    classe: string;
    confianca: number;
    tratamento: string;
  };
};

type Props = NativeStackScreenProps<RootStackParamList, "Preview">;

const API_URL = "http://192.168.15.182:8000";

export function PreviewScreen({ route, navigation }: Props) {
  const { photoUri } = route.params;
  async function sendPhoto(photoUri: string) {
    try {
      const formData = new FormData();
      formData.append("image", {
        uri: photoUri,
        name: "photo1",
        type: "image/jpeg",
      } as any);

      const api = await axios.create({
        baseURL: API_URL,
      });

      const response = await api.postForm("/classificar", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      const data = response.data;

      console.log("✅ Upload sucesso:", data);
      navigation.navigate("Report", {
        photoUri,
        classe: data.classe,
        confianca: data.confianca,
        tratamento: data.tratamento,
      });
    } catch (error: any) {
      console.error("❌ Upload falhou:", error.response?.data || error.message);
      Alert.alert("Erro", error.message);
    }
  }

  function retakePhoto() {
    navigation.goBack();
  }

  return (
    <View style={styles.container}>
      <Image source={{ uri: photoUri }} style={styles.previewImage} />
      <View style={styles.containerImage} />

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={retakePhoto}>
          <MaterialCommunityIcons
            name="camera-retake"
            size={40}
            color="white"
          />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.button}
          onPress={() => sendPhoto(photoUri)}
        >
          <MaterialCommunityIcons name="send" size={40} color="white" />
        </TouchableOpacity>
      </View>
    </View>
  );
}
