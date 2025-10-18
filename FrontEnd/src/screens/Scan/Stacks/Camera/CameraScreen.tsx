import React, { useRef, useState } from "react";
import { View, TouchableOpacity, Alert } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { styles } from "./Styles";
import { NativeStackScreenProps } from "@react-navigation/native-stack";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
};

type Props = NativeStackScreenProps<RootStackParamList, "Camera">;

export function CameraScreen({ navigation }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<"back" | "front">("back");
  const [flash, setFlash] = useState<"off" | "on" | "auto">("off");

  const cameraRef = useRef<any>(null);

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <MaterialCommunityIcons name="camera-off" size={48} color="gray" />
        <TouchableOpacity onPress={requestPermission}>
          <MaterialCommunityIcons name="camera" size={32} color="black" />
        </TouchableOpacity>
      </View>
    );
  }

  // Take photo using camera
  async function takePhoto() {
    if (!cameraRef.current) return;
    const photo = await cameraRef.current.takePictureAsync();
    navigation.navigate("Preview", { photoUri: photo.uri });
  }

  // Pick image from gallery
  async function pickImage() {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      navigation.navigate("Preview", { photoUri: result.assets[0].uri });
    }
  }

  function toggleCameraFacing() {
    setFacing((c) => (c === "back" ? "front" : "back"));
  }

  function toggleFlash() {
    setFlash((c) => (c === "off" ? "on" : c === "on" ? "auto" : "off"));
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
        flash={flash}
      />

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={pickImage}>
          <MaterialCommunityIcons name="image" size={40} color="white" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
          <MaterialCommunityIcons name="camera-flip" size={40} color="white" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.button} onPress={toggleFlash}>
          <MaterialCommunityIcons
            style={{ transform: [{ translateY: 3 }] }}
            name={
              flash === "off"
                ? "flash-off"
                : flash === "on"
                ? "flash"
                : "flash-auto"
            }
            size={40}
            color="white"
          />
        </TouchableOpacity>

        <TouchableOpacity style={styles.button} onPress={takePhoto}>
          <MaterialCommunityIcons name="camera" size={40} color="white" />
        </TouchableOpacity>
      </View>
    </View>
  );
}
